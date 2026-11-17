#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;
namespace py = pybind11;

// this is in the .cu file
void launch_batched_mask_raw(
    int* d_vocab_offsets, char* d_vocab_chars, int vocab_size,
    const vector<string>& batch_texts,
    uintptr_t scores_ptr, int num_beams, int scores_vocab_size, int row_stride, int dtype_code,
    uintptr_t stream_ptr,
    bool* d_is_special, bool* d_is_stop,
    int* d_member_offsets, char* d_member_chars, int member_count,
    int* d_type_offsets, char* d_type_chars, int type_count,
    int* d_case_offsets, char* d_case_chars, int case_count,
    char** d_work_buffer, size_t* work_buffer_size,
    int max_text_per_beam);

// copies a list of words to the gpu
// offsets = where every word starts and ends, chars = all the words behind each other
// if the list is empty the pointers stay NULL and the count is 0
void copy_list_to_gpu(const vector<int>& offsets, const string& chars,
                      int** d_offsets, char** d_chars, int* count)
{
    *d_offsets = NULL;
    *d_chars = NULL;
    *count = 0;
    if (offsets.size() < 2 || chars.empty()) return;

    cudaMalloc((void**)d_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(*d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)d_chars, chars.size() * sizeof(char));
    cudaMemcpy(*d_chars, chars.data(), chars.size() * sizeof(char), cudaMemcpyHostToDevice);

    *count = (int)offsets.size() - 1;
}

class VocabChecker {
public:
    int* d_vocab_offsets;
    char* d_vocab_chars;
    bool* d_is_special;
    bool* d_is_stop;

    int* d_member_offsets;
    char* d_member_chars;
    int member_count;

    int* d_type_offsets;
    char* d_type_chars;
    int type_count;

    int* d_case_offsets;
    char* d_case_chars;
    int case_count;

    char* d_work_buffer;
    size_t work_buffer_size;

    int vocab_size;
    int max_text_per_beam;

    VocabChecker(
        const vector<int>& vocab_offsets, const string& vocab_chars, int v_size,
        const vector<bool>& is_special, const vector<bool>& is_stop,
        const vector<int>& r_recv_offsets, const string& r_recv_chars,
        const vector<int>& r_mem_offsets, const string& r_mem_chars, int n_recv,
        const vector<int>& r_type_offsets, const string& r_type_chars, int n_types,
        const vector<int>& r_case_offsets, const string& r_case_chars, int n_cases)
    {
        d_vocab_offsets = NULL;
        d_vocab_chars = NULL;
        d_is_special = NULL;
        d_is_stop = NULL;
        d_work_buffer = NULL;
        work_buffer_size = 0;
        max_text_per_beam = 4096;
        vocab_size = v_size;

        // vocab
        cudaMalloc((void**)&d_vocab_offsets, vocab_offsets.size() * sizeof(int));
        cudaMemcpy(d_vocab_offsets, vocab_offsets.data(), vocab_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_vocab_chars, vocab_chars.size() * sizeof(char));
        cudaMemcpy(d_vocab_chars, vocab_chars.data(), vocab_chars.size() * sizeof(char), cudaMemcpyHostToDevice);

        // vector<bool> cant be copied directly so make char arrays first
        vector<char> h_special(is_special.size());
        vector<char> h_stop(is_stop.size());
        for (int i = 0; i < (int)is_special.size(); i++) {
            if (is_special[i]) h_special[i] = 1;
            else h_special[i] = 0;
        }
        for (int i = 0; i < (int)is_stop.size(); i++) {
            if (is_stop[i]) h_stop[i] = 1;
            else h_stop[i] = 0;
        }
        cudaMalloc((void**)&d_is_special, h_special.size() * sizeof(bool));
        cudaMemcpy(d_is_special, h_special.data(), h_special.size() * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_is_stop, h_stop.size() * sizeof(bool));
        cudaMemcpy(d_is_stop, h_stop.data(), h_stop.size() * sizeof(bool), cudaMemcpyHostToDevice);

        // the 3 lists (no tries, just arrays)
        // the member list has entries like  receiver<0 char>member
        copy_list_to_gpu(r_mem_offsets, r_mem_chars, &d_member_offsets, &d_member_chars, &member_count);
        copy_list_to_gpu(r_type_offsets, r_type_chars, &d_type_offsets, &d_type_chars, &type_count);
        copy_list_to_gpu(r_case_offsets, r_case_chars, &d_case_offsets, &d_case_chars, &case_count);
    }

    ~VocabChecker() {
        cudaFree(d_vocab_offsets);
        cudaFree(d_vocab_chars);
        cudaFree(d_is_special);
        cudaFree(d_is_stop);

        cudaFree(d_member_offsets);
        cudaFree(d_member_chars);
        cudaFree(d_type_offsets);
        cudaFree(d_type_chars);
        cudaFree(d_case_offsets);
        cudaFree(d_case_chars);

        if (d_work_buffer != NULL) cudaFree(d_work_buffer);
    }

    // masks the scores (python calls this every step)
    void mask_scores_ptr(
        const vector<string>& batch_texts,
        uintptr_t scores_ptr, int num_beams, int scores_vocab_size,
        int row_stride, int dtype_code, uintptr_t stream_ptr)
    {
        launch_batched_mask_raw(
            d_vocab_offsets, d_vocab_chars, vocab_size,
            batch_texts, scores_ptr, num_beams, scores_vocab_size, row_stride, dtype_code, stream_ptr,
            d_is_special, d_is_stop,
            d_member_offsets, d_member_chars, member_count,
            d_type_offsets, d_type_chars, type_count,
            d_case_offsets, d_case_chars, case_count,
            &d_work_buffer, &work_buffer_size,
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
