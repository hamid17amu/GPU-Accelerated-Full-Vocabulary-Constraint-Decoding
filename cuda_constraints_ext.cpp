#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;
namespace py = pybind11;

void launch_batched_mask_raw(
    int* d_vocab_offsets, char* d_vocab_chars, int vocab_size,
    const vector<string>& batch_texts,
    uintptr_t scores_ptr, int num_beams, int scores_vocab_size, int row_stride, int dtype_code,
    uintptr_t stream_ptr,
    bool* d_is_special, bool* d_is_stop,
    int* d_trie_mem_children, bool* d_trie_mem_is_end,
    int* d_trie_type_children, bool* d_trie_type_is_end,
    int* d_trie_case_children, bool* d_trie_case_is_end,
    char** d_batch_text_buf, size_t* batch_text_cap,
    int** d_text_offsets_buf, size_t* text_offsets_cap,
    int max_text_per_beam);

struct TrieNode {
    int children[256];
    bool is_end;

    TrieNode() {
        for (int i = 0; i < 256; ++i) children[i] = -1;
        is_end = false;
    }
};

struct FlattenedTrie {
    vector<int> children;
    vector<char> is_end;
};

FlattenedTrie build_trie(const vector<int>& offsets, const string& chars) {
    FlattenedTrie flat_trie;
    if (offsets.size() < 2 || chars.empty()) return flat_trie;

    vector<TrieNode> nodes;
    nodes.push_back(TrieNode());

    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        int start = offsets[i];
        int end = offsets[i + 1];
        if (start >= end) continue;
        int curr = 0;

        for (int j = start; j < end; ++j) {
            unsigned char c = (unsigned char)chars[j];
            if (nodes[curr].children[c] == -1) {
                nodes[curr].children[c] = (int)nodes.size();
                nodes.push_back(TrieNode());
            }
            curr = nodes[curr].children[c];
        }
        nodes[curr].is_end = true;
    }

    int num_nodes = (int)nodes.size();
    flat_trie.children.resize(num_nodes * 256);
    flat_trie.is_end.resize(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        flat_trie.is_end[i] = nodes[i].is_end ? 1 : 0;
        for (int j = 0; j < 256; ++j) {
            flat_trie.children[i * 256 + j] = nodes[i].children[j];
        }
    }
    return flat_trie;
}

void allocate_trie(const FlattenedTrie& trie, int** d_children, bool** d_is_end) {
    if (trie.children.empty()) {
        *d_children = nullptr;
        *d_is_end = nullptr;
        return;
    }

    cudaMalloc(d_children, trie.children.size() * sizeof(int));
    cudaMemcpy(*d_children, trie.children.data(), trie.children.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(d_is_end, trie.is_end.size() * sizeof(bool));
    cudaMemcpy(*d_is_end, trie.is_end.data(), trie.is_end.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

class VocabChecker {
    int *d_vocab_offsets = nullptr;
    char *d_vocab_chars = nullptr;
    bool *d_is_special = nullptr;
    bool *d_is_stop = nullptr;

    int *d_trie_mem_children = nullptr, *d_trie_type_children = nullptr, *d_trie_case_children = nullptr;
    bool *d_trie_mem_is_end = nullptr, *d_trie_type_is_end = nullptr, *d_trie_case_is_end = nullptr;

    char *d_batch_text_buf = nullptr;
    size_t batch_text_cap = 0;

    int *d_text_offsets_buf = nullptr;
    size_t text_offsets_cap = 0;

    int vocab_size;
    int max_text_per_beam = 4096;

public:
    VocabChecker(
        const vector<int>& vocab_offsets, const string& vocab_chars, int v_size,
        const vector<bool>& is_special, const vector<bool>& is_stop,
        const vector<int>& r_recv_offsets, const string& r_recv_chars,
        const vector<int>& r_mem_offsets, const string& r_mem_chars, int n_recv,
        const vector<int>& r_type_offsets, const string& r_type_chars, int n_types,
        const vector<int>& r_case_offsets, const string& r_case_chars, int n_cases)
        : vocab_size(v_size)
    {
        cudaMalloc(&d_vocab_offsets, vocab_offsets.size() * sizeof(int));
        cudaMemcpy(d_vocab_offsets, vocab_offsets.data(), vocab_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&d_vocab_chars, vocab_chars.size() * sizeof(char));
        cudaMemcpy(d_vocab_chars, vocab_chars.data(), vocab_chars.size() * sizeof(char), cudaMemcpyHostToDevice);

        vector<char> h_special(is_special.size()), h_stop(is_stop.size());
        for (size_t i = 0; i < is_special.size(); ++i) h_special[i] = is_special[i] ? 1 : 0;
        for (size_t i = 0; i < is_stop.size(); ++i) h_stop[i] = is_stop[i] ? 1 : 0;

        cudaMalloc(&d_is_special, h_special.size() * sizeof(bool));
        cudaMemcpy(d_is_special, h_special.data(), h_special.size() * sizeof(bool), cudaMemcpyHostToDevice);

        cudaMalloc(&d_is_stop, h_stop.size() * sizeof(bool));
        cudaMemcpy(d_is_stop, h_stop.data(), h_stop.size() * sizeof(bool), cudaMemcpyHostToDevice);

        allocate_trie(build_trie(r_mem_offsets, r_mem_chars), &d_trie_mem_children, &d_trie_mem_is_end);
        allocate_trie(build_trie(r_type_offsets, r_type_chars), &d_trie_type_children, &d_trie_type_is_end);
        allocate_trie(build_trie(r_case_offsets, r_case_chars), &d_trie_case_children, &d_trie_case_is_end);
    }

    ~VocabChecker() {
        cudaFree(d_vocab_offsets); cudaFree(d_vocab_chars);
        cudaFree(d_is_special); cudaFree(d_is_stop);

        cudaFree(d_trie_mem_children); cudaFree(d_trie_mem_is_end);
        cudaFree(d_trie_type_children); cudaFree(d_trie_type_is_end);
        cudaFree(d_trie_case_children); cudaFree(d_trie_case_is_end);

        if (d_batch_text_buf) cudaFree(d_batch_text_buf);
        if (d_text_offsets_buf) cudaFree(d_text_offsets_buf);
    }

    void mask_scores_ptr(
        const vector<string>& batch_texts,
        uintptr_t scores_ptr, int num_beams, int scores_vocab_size,
        int row_stride, int dtype_code, uintptr_t stream_ptr)
    {
        launch_batched_mask_raw(
            d_vocab_offsets, d_vocab_chars, vocab_size,
            batch_texts, scores_ptr, num_beams, scores_vocab_size, row_stride, dtype_code, stream_ptr,
            d_is_special, d_is_stop,
            d_trie_mem_children, d_trie_mem_is_end,
            d_trie_type_children, d_trie_type_is_end,
            d_trie_case_children, d_trie_case_is_end,
            &d_batch_text_buf, &batch_text_cap,
            &d_text_offsets_buf, &text_offsets_cap,
            max_text_per_beam
        );
    }
};

PYBIND11_MODULE(cuda_constraints, m) {
    py::class_<VocabChecker>(m, "VocabChecker")
        .def(py::init<
            const vector<int>&, const string&, int,
            const vector<bool>&, const vector<bool>&,
            const vector<int>&, const string&,
            const vector<int>&, const string&, int,
            const vector<int>&, const string&, int,
            const vector<int>&, const string&, int>())
        .def("mask_scores_ptr", &VocabChecker::mask_scores_ptr);
}