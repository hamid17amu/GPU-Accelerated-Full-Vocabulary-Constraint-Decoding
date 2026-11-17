// BASELINE gpu version: no trie, no tricks.
// the allowed names are just stored in plain arrays and every thread loops over the arrays.
// one thread = one token. the beams are done one after the other (the cpu loops over them).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>

using namespace std;

// ================== device part (the checking logic) ==================

#define MAX_TOKEN_LENGTH 256

// how the (cleaned) beam text ends, we need this to continue cleaning the token
struct BeamState {
    int textLength;
    int mode;              // 0 = normal code, 1 = inside "string", 2 = inside 'c', 3 = inside // comment, 4 = inside /* comment
    int quoteChar;
    int escapePending;     // 1 if the last char was a backslash
    int slashPending;      // 1 if the last char was a /
    int starPending;       // 1 if the last char was a * (inside a block comment)
};

// letter, number, _ or $
__device__ bool isIdentifierChar(char ch)
{
    if (ch >= 'a' && ch <= 'z') return true;
    if (ch >= 'A' && ch <= 'Z') return true;
    if (ch >= '0' && ch <= '9') return true;
    if (ch == '_' || ch == '$') return true;
    return false;
}

// same but a name cant start with a number
__device__ bool isIdentifierStart(char ch)
{
    if (ch >= 'a' && ch <= 'z') return true;
    if (ch >= 'A' && ch <= 'Z') return true;
    if (ch == '_' || ch == '$') return true;
    return false;
}

// letter, number or _ (no $)
__device__ bool isWordChar(char ch)
{
    if (ch >= 'a' && ch <= 'z') return true;
    if (ch >= 'A' && ch <= 'Z') return true;
    if (ch >= '0' && ch <= '9') return true;
    if (ch == '_') return true;
    return false;
}

__device__ bool isWhitespace(char ch)
{
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v') return true;
    return false;
}

// goes through one char. strings and comments get turned into spaces
// *outChar = the char we keep, *blankPrevious = 1 means the char before has to become a space
__device__ void cleanOneChar(BeamState* state, char ch, char* outChar, int* blankPrevious)
{
    *blankPrevious = 0;

    // normal code
    if (state->mode == 0) {
        if (state->slashPending == 1 && (ch == '/' || ch == '*')) {
            if (ch == '/') state->mode = 3;
            else state->mode = 4;
            state->starPending = 0;
            state->slashPending = 0;
            *blankPrevious = 1;
            *outChar = ' ';
            return;
        }
        state->slashPending = 0;
        if (ch == '"' || ch == '\'') {
            if (ch == '"') state->mode = 1;
            else state->mode = 2;
            state->quoteChar = ch;
            state->escapePending = 0;
            *outChar = ch;
            return;
        }
        if (ch == '/') state->slashPending = 1;
        *outChar = ch;
        return;
    }

    // inside a string or a char
    if (state->mode == 1 || state->mode == 2) {
        if (state->escapePending == 1) {
            state->escapePending = 0;
            *outChar = ' ';
            return;
        }
        if (ch == '\\') {
            state->escapePending = 1;
            *outChar = ' ';
            return;
        }
        if ((int)ch == state->quoteChar) {
            state->mode = 0;
            *outChar = ch;
            return;
        }
        *outChar = ' ';
        return;
    }

    // inside a // comment
    if (state->mode == 3) {
        if (ch == '\n') {
            state->mode = 0;
            *outChar = ch;
            return;
        }
        *outChar = ' ';
        return;
    }

    // inside a /* comment
    if (state->starPending == 1 && ch == '/') {
        state->mode = 0;
        state->starPending = 0;
        *outChar = ' ';
        return;
    }
    if (ch == '*') state->starPending = 1;
    else state->starPending = 0;
    *outChar = ' ';
}

// the cleaned beam text and the cleaned token one after the other
struct JoinedText {
    const char* cleanedBeamText;
    const char* cleanedTokenText;
    int beamTextLength;
    int totalLength;
    int blankLastBeamChar;      // 1 = the last char of the beam text counts as a space
};

__device__ char charAt(JoinedText* text, int position)
{
    if (position < 0 || position >= text->totalLength) return 0;
    if (position < text->beamTextLength) {
        if (text->blankLastBeamChar == 1 && position == text->beamTextLength - 1) return ' ';
        return text->cleanedBeamText[position];
    }
    return text->cleanedTokenText[position - text->beamTextLength];
}

// all the arrays we need
// every list is stored as: offsets (where word number i starts and ends) + one long char array
// the member list has entries like  receiver<0 char>member   (for example  foo<0>getRow)
struct AllData {
    const int* vocabOffsets;
    const char* vocabChars;
    const bool* isSpecialToken;
    const bool* isStopToken;
    const char* cleanedTexts;
    const int* textOffsets;
    const BeamState* beamStates;
    int vocabSize;

    const int* memberOffsets;  const char* memberChars;  int memberCount;
    const int* typeOffsets;    const char* typeChars;    int typeCount;
    const int* caseOffsets;    const char* caseChars;    int caseCount;
};

// finds the text in front of a dot (like foo or this.plot or getPlot())
// gives back its start and end in *receiverStart and *receiverEnd, returns 0 if there is none
__device__ int findReceiver(JoinedText* text, int dotPosition, int* receiverStart, int* receiverEnd)
{
    int i = dotPosition - 1;
    while (i >= 0 && isWhitespace(charAt(text, i))) i--;
    int end = i + 1;

    while (i >= 0) {
        if (charAt(text, i) == ')') {
            // skip everything inside the brackets
            int depth = 0;
            while (i >= 0) {
                char ch = charAt(text, i);
                if (ch == ')') depth++;
                else if (ch == '(') {
                    depth--;
                    if (depth == 0) break;
                }
                i--;
            }
            i--;
        }
        int j = i;
        while (j >= 0 && isIdentifierChar(charAt(text, j))) j--;
        if (j == i) break;
        i = j;
        int k = i;
        while (k >= 0 && isWhitespace(charAt(text, k))) k--;
        if (k >= 0 && charAt(text, k) == '.') {
            // another dot before, keep going back
            i = k - 1;
            while (i >= 0 && isWhitespace(charAt(text, i))) i--;
            continue;
        }
        break;
    }

    int start = i + 1;
    while (start < end && isWhitespace(charAt(text, start))) start++;
    if (start >= end) return 0;
    char firstChar = charAt(text, start);
    if (firstChar >= '0' && firstChar <= '9') return 0;

    *receiverStart = start;
    *receiverEnd = end;
    return 1;
}

// compares the name in the text with one word from a list
// mustBeComplete = 1: the word has to be exactly the name
// mustBeComplete = 0: the name only has to be the beginning of the word
__device__ int nameMatchesWord(JoinedText* text, int nameStart, int nameEnd,
                               const char* listChars, int wordStart, int wordEnd, int mustBeComplete)
{
    int nameLength = nameEnd - nameStart;
    int wordLength = wordEnd - wordStart;
    if (mustBeComplete == 1) {
        if (nameLength != wordLength) return 0;
    } else {
        if (nameLength > wordLength) return 0;
    }
    for (int i = 0; i < nameLength; i++) {
        if (charAt(text, nameStart + i) != listChars[wordStart + i]) return 0;
    }
    return 1;
}

// is the name ok for this receiver? loops over the whole member list
// if the receiver is not in the list at all we dont know it, so everything is ok
__device__ int memberNameIsOk(JoinedText* text, int receiverStart, int receiverEnd,
                              int nameStart, int nameEnd, int nameIsComplete, AllData* data)
{
    int receiverWasFound = 0;

    for (int entry = 0; entry < data->memberCount; entry++) {
        int entryStart = data->memberOffsets[entry];
        int entryEnd = data->memberOffsets[entry + 1];
        if (entryStart >= entryEnd) continue;

        // find the 0 char, it splits the receiver and the member
        int zeroPosition = entryStart;
        while (zeroPosition < entryEnd && data->memberChars[zeroPosition] != 0) zeroPosition++;
        if (zeroPosition >= entryEnd) continue;

        // is the receiver of this entry the same as the receiver in the text? (spaces in the text are skipped)
        int sameReceiver = 1;
        int entryPos = entryStart;
        for (int p = receiverStart; p < receiverEnd; p++) {
            char ch = charAt(text, p);
            if (isWhitespace(ch)) continue;
            if (entryPos >= zeroPosition || data->memberChars[entryPos] != ch) {
                sameReceiver = 0;
                break;
            }
            entryPos++;
        }
        if (sameReceiver == 1 && entryPos != zeroPosition) sameReceiver = 0;
        if (sameReceiver == 0) continue;

        receiverWasFound = 1;
        if (nameMatchesWord(text, nameStart, nameEnd, data->memberChars, zeroPosition + 1, entryEnd, nameIsComplete)) return 1;
    }

    if (receiverWasFound == 0) return 1;
    return 0;
}

// is the name in a plain list (types or case constants)? loops over the whole list
__device__ int nameIsInList(JoinedText* text, int nameStart, int nameEnd, int nameIsComplete,
                            const int* listOffsets, const char* listChars, int listCount)
{
    for (int entry = 0; entry < listCount; entry++) {
        int entryStart = listOffsets[entry];
        int entryEnd = listOffsets[entry + 1];
        if (entryStart >= entryEnd) continue;
        if (nameMatchesWord(text, nameStart, nameEnd, listChars, entryStart, entryEnd, nameIsComplete)) return 1;
    }
    return 0;
}

// is the word at this position
__device__ bool wordMatchesAt(JoinedText* text, int position, const char* word, int wordLength)
{
    for (int i = 0; i < wordLength; i++) {
        if (charAt(text, position + i) != word[i]) return false;
    }
    return true;
}

// checks every dot / new / case that touches the end of the old beam text
// (the older ones were already checked in earlier steps)
__device__ int checkAllSites(JoinedText* text, int finalCheck, int textEndsInLiteral, AllData* data)
{
    int beamTextLength = text->beamTextLength;
    int totalLength = text->totalLength;

    // go back a bit to find where we have to start looking
    int startSearch = beamTextLength;
    while (startSearch > 0 && isIdentifierChar(charAt(text, startSearch - 1))) startSearch--;
    while (startSearch > 0 && isWhitespace(charAt(text, startSearch - 1))) startSearch--;
    while (startSearch > 0 && isIdentifierChar(charAt(text, startSearch - 1))) startSearch--;
    int firstPosition = 0;
    if (startSearch > 0) firstPosition = startSearch - 1;

    for (int pos = firstPosition; pos < totalLength; pos++) {
        char ch = charAt(text, pos);

        if (ch == '.') {
            // a member after a dot
            if (data->memberCount == 0) continue;
            int nameStart = pos + 1;
            while (nameStart < totalLength && isWhitespace(charAt(text, nameStart))) nameStart++;
            if (nameStart >= totalLength || !isIdentifierStart(charAt(text, nameStart))) continue;
            int nameEnd = nameStart;
            while (nameEnd < totalLength && isIdentifierChar(charAt(text, nameEnd))) nameEnd++;
            if (nameEnd < beamTextLength) continue;     // this name was already checked before

            int receiverStart = 0;
            int receiverEnd = 0;
            if (findReceiver(text, pos, &receiverStart, &receiverEnd) == 0) continue;

            int nameIsComplete = 0;
            if (nameEnd < totalLength || finalCheck == 1 || textEndsInLiteral == 1) nameIsComplete = 1;
            if (memberNameIsOk(text, receiverStart, receiverEnd, nameStart, nameEnd, nameIsComplete, data) == 0) return 0;
        }
        else if (ch == 'c' || ch == 'n') {
            // maybe the word case or the word new
            int isCase = 0;
            if (ch == 'c') isCase = 1;
            int keywordLength = 3;
            if (isCase == 1) keywordLength = 4;

            int listCount = data->typeCount;
            if (isCase == 1) listCount = data->caseCount;
            if (listCount == 0) continue;

            if (pos > 0 && isWordChar(charAt(text, pos - 1))) continue;    // part of a bigger word
            if (isCase == 1) {
                if (!wordMatchesAt(text, pos, "case", 4)) continue;
            } else {
                if (!wordMatchesAt(text, pos, "new", 3)) continue;
            }
            int nameStart = pos + keywordLength;
            if (!(nameStart < totalLength && isWhitespace(charAt(text, nameStart)))) continue;
            while (nameStart < totalLength && isWhitespace(charAt(text, nameStart))) nameStart++;
            if (nameStart >= totalLength || !isIdentifierStart(charAt(text, nameStart))) continue;
            int nameEnd = nameStart;
            while (nameEnd < totalLength && isIdentifierChar(charAt(text, nameEnd))) nameEnd++;
            if (nameEnd < beamTextLength) continue;

            if (isCase == 0) {
                // new a.b.C has a dot after the name, those are skipped
                int afterName = nameEnd;
                while (afterName < totalLength && isWhitespace(charAt(text, afterName))) afterName++;
                if (afterName < totalLength && charAt(text, afterName) == '.') continue;
            }

            int nameIsComplete = 0;
            if (nameEnd < totalLength || finalCheck == 1 || textEndsInLiteral == 1) nameIsComplete = 1;

            int ok = 0;
            if (isCase == 1) ok = nameIsInList(text, nameStart, nameEnd, nameIsComplete, data->caseOffsets, data->caseChars, data->caseCount);
            else ok = nameIsInList(text, nameStart, nameEnd, nameIsComplete, data->typeOffsets, data->typeChars, data->typeCount);
            if (ok == 0) return 0;
        }
    }
    return 1;
}

// cleans the text of one beam (strings and comments become spaces) and saves how it ends
__device__ void prepareBeam(const char* beamText, int length, char* cleanedText, BeamState* state)
{
    state->textLength = length;
    state->mode = 0;
    state->quoteChar = 0;
    state->escapePending = 0;
    state->slashPending = 0;
    state->starPending = 0;

    for (int i = 0; i < length; i++) {
        char outChar;
        int blankPrevious;
        cleanOneChar(state, beamText[i], &outChar, &blankPrevious);
        cleanedText[i] = outChar;
        if (blankPrevious == 1 && i > 0) cleanedText[i - 1] = ' ';
    }
}

// is this token allowed after this beam text (this is the whole check)
__device__ int isTokenAllowed(AllData* data, int beam, int tokenId)
{
    int isStop = 0;
    if (data->isStopToken[tokenId]) isStop = 1;
    int isSpecial = 0;
    if (data->isSpecialToken[tokenId]) isSpecial = 1;

    if (isSpecial == 1 && isStop == 0) return 1;      // special tokens are always ok

    // nothing to check if we have no lists at all
    if (data->memberCount == 0 && data->typeCount == 0 && data->caseCount == 0) return 1;

    BeamState state = data->beamStates[beam];
    const char* cleanedBeamText = data->cleanedTexts + data->textOffsets[beam];

    // a stop token adds no text, we only check that the text can end here
    int tokenStart = data->vocabOffsets[tokenId];
    int tokenLength = data->vocabOffsets[tokenId + 1] - tokenStart;
    if (isStop == 1) tokenLength = 0;
    if (tokenLength > MAX_TOKEN_LENGTH) tokenLength = MAX_TOKEN_LENGTH;

    // clean the token, continuing where the beam text stopped
    char cleanedToken[MAX_TOKEN_LENGTH];
    int blankLastBeamChar = 0;
    for (int k = 0; k < tokenLength; k++) {
        char outChar;
        int blankPrevious;
        cleanOneChar(&state, data->vocabChars[tokenStart + k], &outChar, &blankPrevious);
        cleanedToken[k] = outChar;
        if (blankPrevious == 1) {
            if (k == 0) blankLastBeamChar = 1;
            else cleanedToken[k - 1] = ' ';
        }
    }

    JoinedText text;
    text.cleanedBeamText = cleanedBeamText;
    text.cleanedTokenText = cleanedToken;
    text.beamTextLength = data->beamStates[beam].textLength;
    text.totalLength = text.beamTextLength + tokenLength;
    text.blankLastBeamChar = blankLastBeamChar;

    int textEndsInLiteral = 0;
    if (state.mode != 0) textEndsInLiteral = 1;
    return checkAllSites(&text, isStop, textEndsInLiteral, data);
}

// ================== end of device part ==================


// kernel 1: cleans the text of ONE beam (only one thread does the work)
__global__ void prepareBeamKernel(AllData data, int beam, const char* beamTexts, char* cleanedTexts, BeamState* beamStates)
{
    int textStart = data.textOffsets[beam];
    int textLength = data.textOffsets[beam + 1] - textStart;
    BeamState state;
    prepareBeam(beamTexts + textStart, textLength, cleanedTexts + textStart, &state);
    beamStates[beam] = state;
}

// kernel 2: one thread for every token of ONE beam
// writes 1 or 0 into allowedFlags and counts the allowed tokens of the beam
__global__ void judgeTokensKernel(AllData data, int beam, int scoresWidth, unsigned char* allowedFlags, int* allowedCounts)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tokenId >= scoresWidth) return;

    int allowed = 0;
    if (tokenId < data.vocabSize) allowed = isTokenAllowed(&data, beam, tokenId);   // ids after the vocab are never allowed

    allowedFlags[beam * scoresWidth + tokenId] = allowed;
    if (allowed == 1) atomicAdd(&allowedCounts[beam], 1);
}

// kernel 3: put -inf on every token of ONE beam that is not allowed
// if the beam has no allowed token at all we leave it alone
// (float version)
__global__ void applyMaskFloat(float* scores, const unsigned char* allowedFlags, const int* allowedCounts, int beam, int scoresWidth, int rowStride)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = -INFINITY;
    }
}

// (half version)
__global__ void applyMaskHalf(__half* scores, const unsigned char* allowedFlags, const int* allowedCounts, int beam, int scoresWidth, int rowStride)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = __float2half(-INFINITY);
    }
}

// (bfloat16 version)
__global__ void applyMaskBf16(__nv_bfloat16* scores, const unsigned char* allowedFlags, const int* allowedCounts, int beam, int scoresWidth, int rowStride)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = __float2bfloat16(-INFINITY);
    }
}

// dtype_code: 0 = float32, 1 = float16, 2 = bfloat16
// the cpp file calls this every decoding step
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
    int max_text_per_beam)
{
    if (num_beams == 0 || scores_ptr == 0) return;

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int scoresWidth = scores_vocab_size;

    // put the texts of all beams behind each other (we only keep the last max_text_per_beam chars of each)
    vector<int> textOffsets(num_beams + 1, 0);
    string allTexts;
    for (int beam = 0; beam < num_beams; beam++) {
        int length = 0;
        if (beam < (int)batch_texts.size()) {
            const string& text = batch_texts[beam];
            int startAt = 0;
            if ((int)text.length() > max_text_per_beam) startAt = (int)text.length() - max_text_per_beam;
            length = (int)text.length() - startAt;
            allTexts.append(text.data() + startAt, length);
        }
        textOffsets[beam + 1] = textOffsets[beam] + length;
    }
    int totalTextChars = (int)allTexts.size();

    // where everything goes inside one big gpu buffer (every part starts at a multiple of 16)
    int offsetsStart = 0;
    int textsStart = ((num_beams + 1) * 4 + 15) / 16 * 16;
    int cleanedTextsStart = (textsStart + totalTextChars + 15) / 16 * 16;
    int statesStart = (cleanedTextsStart + totalTextChars + 15) / 16 * 16;
    int countsStart = (statesStart + num_beams * (int)sizeof(BeamState) + 15) / 16 * 16;
    int flagsStart = (countsStart + num_beams * 4 + 15) / 16 * 16;
    int bytesNeeded = flagsStart + num_beams * scoresWidth + 16;

    // make the buffer bigger if its too small
    if (*d_work_buffer == NULL || (size_t)bytesNeeded > *work_buffer_size) {
        if (*d_work_buffer != NULL) cudaFree(*d_work_buffer);
        *work_buffer_size = (size_t)bytesNeeded * 2;
        cudaMalloc((void**)d_work_buffer, *work_buffer_size);
    }
    char* gpuBase = *d_work_buffer;

    // copy the offsets and the texts to the gpu
    cudaMemcpyAsync(gpuBase + offsetsStart, textOffsets.data(), (num_beams + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (totalTextChars > 0) {
        cudaMemcpyAsync(gpuBase + textsStart, allTexts.data(), totalTextChars, cudaMemcpyHostToDevice, stream);
    }
    int* gpuAllowedCounts = (int*)(gpuBase + countsStart);
    cudaMemsetAsync(gpuAllowedCounts, 0, num_beams * sizeof(int), stream);

    AllData data;
    data.vocabOffsets = d_vocab_offsets;
    data.vocabChars = d_vocab_chars;
    data.isSpecialToken = d_is_special;
    data.isStopToken = d_is_stop;
    data.cleanedTexts = gpuBase + cleanedTextsStart;
    data.textOffsets = (const int*)(gpuBase + offsetsStart);
    data.beamStates = (const BeamState*)(gpuBase + statesStart);
    data.vocabSize = vocab_size;
    data.memberOffsets = d_member_offsets;
    data.memberChars = d_member_chars;
    data.memberCount = member_count;
    data.typeOffsets = d_type_offsets;
    data.typeChars = d_type_chars;
    data.typeCount = type_count;
    data.caseOffsets = d_case_offsets;
    data.caseChars = d_case_chars;
    data.caseCount = case_count;

    unsigned char* gpuAllowedFlags = (unsigned char*)(gpuBase + flagsStart);
    int threads = 256;
    int blocks = (scoresWidth + 255) / 256;

    // go through the beams one after the other
    for (int beam = 0; beam < num_beams; beam++) {

        // step 1: clean the text of this beam
        prepareBeamKernel<<<1, 1, 0, stream>>>(
            data, beam, gpuBase + textsStart, gpuBase + cleanedTextsStart, (BeamState*)(gpuBase + statesStart));

        // step 2: check every token for this beam
        judgeTokensKernel<<<blocks, threads, 0, stream>>>(data, beam, scoresWidth, gpuAllowedFlags, gpuAllowedCounts);

        // step 3: put -inf on the tokens that are not allowed
        if (dtype_code == 0) {
            applyMaskFloat<<<blocks, threads, 0, stream>>>(
                (float*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, beam, scoresWidth, row_stride);
        }
        else if (dtype_code == 1) {
            applyMaskHalf<<<blocks, threads, 0, stream>>>(
                (__half*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, beam, scoresWidth, row_stride);
        }
        else {
            applyMaskBf16<<<blocks, threads, 0, stream>>>(
                (__nv_bfloat16*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, beam, scoresWidth, row_stride);
        }
    }
}
