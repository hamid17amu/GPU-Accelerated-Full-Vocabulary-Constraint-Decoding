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

    // everything we know about the end of one beam text
    // prepareBeam fills this in
struct BeamState {
    int textLength;          // how many chars the beam text has
    int mode;                // 0 = normal code, 1 = inside "string", 2 = inside 'c', 3 = inside // comment, 4 = inside /* comment
    int quoteChar;           // the quote char of the string we are in
    int escapePending;       // 1 if the last char was a backslash
    int slashPending;        // 1 if the last char was a /
    int starPending;         // 1 if the last char was a * (inside a block comment)
    int stopTokenAllowed;    // 1 if the text is allowed to end here
    int siteKind;            // what name is open at the end: 0 none, 1 member (after dot), 2 type (after new), 3 case
    int trieNode;            // where we are in the trie for the name typed so far
    int nameLength;          // how many letters of the name are typed
    int keywordCode;         // last word is the start of a keyword: 1 n, 2 ne, 3 c, 4 ca, 5 cas, 6 new, 7 case
    int forceFullCheck;      // 1 = always do the slow check for this beam
    int pad0, pad1;
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

// same as above but a name cant start with a number
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

// goes through one char of the text. strings and comments get turned into spaces
// so that stuff inside them is not checked
// *outChar = the char we keep, *blankPrevious = 1 means the char before has to become a space
__device__ void cleanOneChar(BeamState* state, char ch, char* outChar, int* blankPrevious)
{
    *blankPrevious = 0;

    // normal code
    if (state->mode == 0) {
        if (state->slashPending == 1 && (ch == '/' || ch == '*')) {
            // the / before and this char start a comment
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
            // string is over
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
// (this way we dont have to copy them into one big array)
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

// looks at what is in front of a dot (like foo or this.plot or getPlot())
// and finds it in the member trie. gives back the trie node or -1 if its not known
__device__ int findReceiverNode(JoinedText* text, int dotPosition, const int* memberTrieChildren)
{
    int i = dotPosition - 1;
    while (i >= 0 && isWhitespace(charAt(text, i))) i--;
    int receiverEnd = i + 1;

    while (i >= 0) {
        if (charAt(text, i) == ')') {
            // skip the stuff inside the brackets
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
            // there is another dot before, so keep going back
            i = k - 1;
            while (i >= 0 && isWhitespace(charAt(text, i))) i--;
            continue;
        }
        break;
    }

    int receiverStart = i + 1;
    while (receiverStart < receiverEnd && isWhitespace(charAt(text, receiverStart))) receiverStart++;
    if (receiverStart >= receiverEnd) return -1;
    char firstChar = charAt(text, receiverStart);
    if (firstChar >= '0' && firstChar <= '9') return -1;

    // walk the receiver through the trie
    int node = 0;
    for (int pos = receiverStart; pos < receiverEnd; pos++) {
        char ch = charAt(text, pos);
        if (isWhitespace(ch)) continue;
        node = memberTrieChildren[node * 256 + (unsigned char)ch];
        if (node < 0) return -1;
    }
    // after the receiver there must be a 0 char, then come the members
    return memberTrieChildren[node * 256 + 0];
}

// walks the name (from nameStart to nameEnd) through the trie
// if the name is complete it also has to be the end of a word in the trie
__device__ bool nameIsOk(const int* trieChildren, const bool* trieIsEnd, int rootNode,
                         JoinedText* text, int nameStart, int nameEnd, bool nameIsComplete)
{
    int node = rootNode;
    for (int i = nameStart; i < nameEnd; i++) {
        node = trieChildren[node * 256 + (unsigned char)charAt(text, i)];
        if (node < 0) return false;
    }
    if (nameIsComplete) return trieIsEnd[node];
    return true;
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
__device__ bool checkAllSites(JoinedText* text, bool finalCheck, bool textEndsInLiteral,
    const int* memberTrieChildren, const bool* memberTrieIsEnd,
    const int* typeTrieChildren, const bool* typeTrieIsEnd,
    const int* caseTrieChildren, const bool* caseTrieIsEnd)
{
    int beamTextLength = text->beamTextLength;
    int totalLength = text->totalLength;

    // go back a little bit to find where we have to start looking
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
            if (memberTrieChildren == NULL) continue;
            int nameStart = pos + 1;
            while (nameStart < totalLength && isWhitespace(charAt(text, nameStart))) nameStart++;
            if (nameStart >= totalLength || !isIdentifierStart(charAt(text, nameStart))) continue;
            int nameEnd = nameStart;
            while (nameEnd < totalLength && isIdentifierChar(charAt(text, nameEnd))) nameEnd++;
            if (nameEnd < beamTextLength) continue;    // this name was already checked before
            int receiverNode = findReceiverNode(text, pos, memberTrieChildren);
            if (receiverNode < 0) continue;            // we dont know this receiver so everything is ok
            bool nameIsComplete = false;
            if (nameEnd < totalLength || finalCheck || textEndsInLiteral) nameIsComplete = true;
            if (!nameIsOk(memberTrieChildren, memberTrieIsEnd, receiverNode, text, nameStart, nameEnd, nameIsComplete)) return false;
        }
        else if (ch == 'c' || ch == 'n') {
            // maybe the word case or the word new
            bool isCase = false;
            if (ch == 'c') isCase = true;
            int keywordLength = 3;
            if (isCase) keywordLength = 4;
            const int* trieChildren = typeTrieChildren;
            const bool* trieIsEnd = typeTrieIsEnd;
            if (isCase) {
                trieChildren = caseTrieChildren;
                trieIsEnd = caseTrieIsEnd;
            }
            if (trieChildren == NULL) continue;
            if (pos > 0 && isWordChar(charAt(text, pos - 1))) continue;     // part of a bigger word
            if (isCase) {
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
            if (isCase == false) {
                // new a.b.C has a dot after the name, we skip those
                int afterName = nameEnd;
                while (afterName < totalLength && isWhitespace(charAt(text, afterName))) afterName++;
                if (afterName < totalLength && charAt(text, afterName) == '.') continue;
            }
            bool nameIsComplete = false;
            if (nameEnd < totalLength || finalCheck || textEndsInLiteral) nameIsComplete = true;
            if (!nameIsOk(trieChildren, trieIsEnd, 0, text, nameStart, nameEnd, nameIsComplete)) return false;
        }
    }
    return true;
}

// all the gpu pointers we need, in one struct so we can pass it around
struct AllData {
    const int* vocabOffsets;
    const char* vocabChars;
    const bool* isSpecialToken;
    const bool* isStopToken;
    const char* cleanedTexts;
    const int* textOffsets;
    const BeamState* beamStates;
    int vocabSize;
    const int* memberTrieChildren;  const bool* memberTrieIsEnd;
    const int* typeTrieChildren;    const bool* typeTrieIsEnd;
    const int* caseTrieChildren;    const bool* caseTrieIsEnd;
};

// true if we have no tries at all (then nothing is checked)
__device__ bool noTriesExist(AllData* data)
{
    if (data->memberTrieChildren == NULL && data->typeTrieChildren == NULL && data->caseTrieChildren == NULL) return true;
    return false;
}

// runs one time for every beam: cleans the text and works out how the text ends
__device__ void prepareBeam(AllData* data, const char* beamText, int length, char* cleanedText, BeamState* state)
{
    state->textLength = length;
    state->mode = 0;
    state->quoteChar = 0;
    state->escapePending = 0;
    state->slashPending = 0;
    state->starPending = 0;
    state->stopTokenAllowed = 1;
    state->siteKind = 0;
    state->trieNode = 0;
    state->nameLength = 0;
    state->keywordCode = 0;
    state->forceFullCheck = 0;
    state->pad0 = 0;
    state->pad1 = 0;

    // remove the strings and the comments
    for (int i = 0; i < length; i++) {
        char outChar;
        int blankPrevious;
        cleanOneChar(state, beamText[i], &outChar, &blankPrevious);
        cleanedText[i] = outChar;
        if (blankPrevious == 1 && i > 0) cleanedText[i - 1] = ' ';
    }

    if (noTriesExist(data)) return;

    JoinedText text;
    text.cleanedBeamText = cleanedText;
    text.cleanedTokenText = cleanedText;
    text.beamTextLength = length;
    text.totalLength = length;
    text.blankLastBeamChar = 0;

    // can a stop token come now? (every name at the end has to be complete)
    bool textEndsInLiteral = false;
    if (state->mode != 0) textEndsInLiteral = true;
    bool stopIsOk = checkAllSites(&text, true, textEndsInLiteral,
        data->memberTrieChildren, data->memberTrieIsEnd,
        data->typeTrieChildren, data->typeTrieIsEnd,
        data->caseTrieChildren, data->caseTrieIsEnd);
    if (stopIsOk) state->stopTokenAllowed = 1;
    else state->stopTokenAllowed = 0;

    if (state->mode != 0) return;

    // the word at the very end of the text
    int wordStart = length;
    while (wordStart > 0 && isIdentifierChar(cleanedText[wordStart - 1])) wordStart--;
    int wordLength = length - wordStart;
    state->nameLength = wordLength;

    // is this word the start of new or case?
    if (wordLength == 1) {
        if (cleanedText[wordStart] == 'n') state->keywordCode = 1;
        else if (cleanedText[wordStart] == 'c') state->keywordCode = 3;
    }
    else if (wordLength == 2) {
        if (cleanedText[wordStart] == 'n' && cleanedText[wordStart + 1] == 'e') state->keywordCode = 2;
        else if (cleanedText[wordStart] == 'c' && cleanedText[wordStart + 1] == 'a') state->keywordCode = 4;
    }
    else if (wordLength == 3) {
        if (cleanedText[wordStart] == 'c' && cleanedText[wordStart + 1] == 'a' && cleanedText[wordStart + 2] == 's') state->keywordCode = 5;
        else if (cleanedText[wordStart] == 'n' && cleanedText[wordStart + 1] == 'e' && cleanedText[wordStart + 2] == 'w') state->keywordCode = 6;
    }
    else if (wordLength == 4) {
        if (cleanedText[wordStart] == 'c' && cleanedText[wordStart + 1] == 'a' && cleanedText[wordStart + 2] == 's' && cleanedText[wordStart + 3] == 'e') state->keywordCode = 7;
    }

    // a word that starts with a number is not a name
    if (wordLength > 0 && !isIdentifierStart(cleanedText[wordStart])) return;

    // what is in front of the word? (spaces and then a dot, new or case)
    int beforeWord = wordStart - 1;
    while (beforeWord >= 0 && isWhitespace(cleanedText[beforeWord])) beforeWord--;
    int siteKind = 0;
    int rootNode = -1;

    if (beforeWord >= 0 && cleanedText[beforeWord] == '.') {
        if (data->memberTrieChildren != NULL) {
            rootNode = findReceiverNode(&text, beforeWord, data->memberTrieChildren);
            if (rootNode >= 0) siteKind = 1;
        }
    }
    else if (beforeWord >= 0 && beforeWord < wordStart - 1) {
        // there was at least one space, so maybe new or case is in front
        int keywordStart = beforeWord;
        while (keywordStart >= 0 && isIdentifierChar(cleanedText[keywordStart])) keywordStart--;
        int keywordLength = beforeWord - keywordStart;
        if (keywordLength == 3 && cleanedText[keywordStart + 1] == 'n' && cleanedText[keywordStart + 2] == 'e' &&
            cleanedText[keywordStart + 3] == 'w' && data->typeTrieChildren != NULL) {
            siteKind = 2;
            rootNode = 0;
        }
        else if (keywordLength == 4 && cleanedText[keywordStart + 1] == 'c' && cleanedText[keywordStart + 2] == 'a' &&
                 cleanedText[keywordStart + 3] == 's' && cleanedText[keywordStart + 4] == 'e' && data->caseTrieChildren != NULL) {
            siteKind = 3;
            rootNode = 0;
        }
    }
    if (siteKind == 0) return;

    const int* trieChildren = data->memberTrieChildren;
    if (siteKind == 2) trieChildren = data->typeTrieChildren;
    if (siteKind == 3) trieChildren = data->caseTrieChildren;

    // walk the part of the name that is already typed
    int node = rootNode;
    for (int i = wordStart; i < length; i++) {
        node = trieChildren[node * 256 + (unsigned char)cleanedText[i]];
        if (node < 0) {
            state->forceFullCheck = 1;
            break;
        }
    }
    state->siteKind = siteKind;
    state->trieNode = node;
}

#define FLAG_HAS_DOT 1
#define FLAG_HAS_QUOTE 2
#define FLAG_HAS_SLASH 4
#define FLAG_HAS_NEW_OR_CASE 8
#define FLAG_HAS_NEWLINE 16
#define FLAG_TOO_LONG 32

// info about one token
struct TokenInfo {
    const char* tokenChars;
    int length;
    int leadingIdentifierLength;    // how many identifier chars the token starts with
    int flags;                      // the FLAG_ things
    int tokenClass;                 // 0 normal, 1 stop token, 2 special token
    unsigned char firstChar;
};

// runs one time for every token
__device__ TokenInfo getTokenInfo(AllData* data, int tokenId)
{
    TokenInfo info;
    info.tokenClass = 0;
    if (data->isSpecialToken[tokenId]) info.tokenClass = 2;
    if (data->isStopToken[tokenId]) info.tokenClass = 1;

    int start = data->vocabOffsets[tokenId];
    int end = data->vocabOffsets[tokenId + 1];
    info.tokenChars = data->vocabChars + start;
    info.length = end - start;
    info.flags = 0;
    info.leadingIdentifierLength = 0;
    info.firstChar = 0;
    if (info.length > MAX_TOKEN_LENGTH) info.flags = info.flags | FLAG_TOO_LONG;
    if (info.length > 0) info.firstChar = (unsigned char)info.tokenChars[0];

    int stillAtStart = 1;
    for (int i = 0; i < info.length; i++) {
        char ch = info.tokenChars[i];
        if (stillAtStart == 1 && isIdentifierChar(ch)) info.leadingIdentifierLength++;
        else stillAtStart = 0;

        if (ch == '.') info.flags = info.flags | FLAG_HAS_DOT;
        else if (ch == '"' || ch == '\'') info.flags = info.flags | FLAG_HAS_QUOTE;
        else if (ch == '/') info.flags = info.flags | FLAG_HAS_SLASH;
        else if (ch == '\n') info.flags = info.flags | FLAG_HAS_NEWLINE;
        else if (ch == 'w' && i >= 2 && info.tokenChars[i - 1] == 'e' && info.tokenChars[i - 2] == 'n') info.flags = info.flags | FLAG_HAS_NEW_OR_CASE;
        else if (ch == 'e' && i >= 3 && info.tokenChars[i - 1] == 's' && info.tokenChars[i - 2] == 'a' && info.tokenChars[i - 3] == 'c') info.flags = info.flags | FLAG_HAS_NEW_OR_CASE;
    }
    return info;
}

// the quick check. gives 1 = allowed, 0 = not allowed, 2 = dont know (we need the full check)
__device__ int quickDecision(AllData* data, BeamState* state, TokenInfo* token)
{
    if (token->flags & FLAG_TOO_LONG) return 2;

    // the beam text ends inside a string or a comment
    if (state->mode != 0) {
        if (state->mode == 1 || state->mode == 2) {
            if (token->flags & FLAG_HAS_QUOTE) return 2;
            return 1;
        }
        if (state->mode == 3) {
            if (token->flags & FLAG_HAS_NEWLINE) return 2;
            return 1;
        }
        if (token->flags & FLAG_HAS_SLASH) return 2;
        return 1;
    }

    if (state->forceFullCheck == 1) return 2;
    if (token->flags & (FLAG_HAS_DOT | FLAG_HAS_QUOTE | FLAG_HAS_SLASH | FLAG_HAS_NEW_OR_CASE)) return 2;
    if (token->length == 0) return 1;
    if (state->slashPending == 1 && token->firstChar == '*') return 2;

    // the token could finish the word new or case
    if (state->keywordCode >= 1 && state->keywordCode <= 5) {
        const char* missingPart = "e";
        int missingLength = 1;
        if (state->keywordCode == 1) { missingPart = "ew"; missingLength = 2; }
        if (state->keywordCode == 2) { missingPart = "w"; missingLength = 1; }
        if (state->keywordCode == 3) { missingPart = "ase"; missingLength = 3; }
        if (state->keywordCode == 4) { missingPart = "se"; missingLength = 2; }
        if (state->keywordCode == 5) { missingPart = "e"; missingLength = 1; }
        if (token->length >= missingLength) {
            int sameAsMissingPart = 1;
            for (int i = 0; i < missingLength; i++) {
                if (token->tokenChars[i] != missingPart[i]) {
                    sameAsMissingPart = 0;
                    break;
                }
            }
            if (sameAsMissingPart == 1) return 2;
        }
    }
    else if (state->keywordCode >= 6) {
        // the text ends with new or case
        if (isWhitespace((char)token->firstChar) || isIdentifierChar((char)token->firstChar)) return 2;
    }

    // no name is open at the end, so nothing to check
    if (state->siteKind == 0) return 1;

    char firstChar = (char)token->firstChar;
    if (state->nameLength > 0) {
        // the name is already started
        if (!isIdentifierChar(firstChar)) {
            // token ends the name, so the name has to be complete
            if (state->stopTokenAllowed) return 1;
            return 0;
        }
    }
    else {
        // no name yet
        if (isWhitespace(firstChar)) return 2;
        if (!isIdentifierChar(firstChar)) return 1;
        if (!isIdentifierStart(firstChar)) return 2;
    }

    const int* trieChildren = data->memberTrieChildren;
    const bool* trieIsEnd = data->memberTrieIsEnd;
    if (state->siteKind == 2) { trieChildren = data->typeTrieChildren; trieIsEnd = data->typeTrieIsEnd; }
    if (state->siteKind == 3) { trieChildren = data->caseTrieChildren; trieIsEnd = data->caseTrieIsEnd; }

    // continue the name in the trie
    int node = state->trieNode;
    for (int i = 0; i < token->leadingIdentifierLength; i++) {
        node = trieChildren[node * 256 + (unsigned char)token->tokenChars[i]];
        if (node < 0) return 0;
    }
    if (token->leadingIdentifierLength == token->length) return 1;   // name is not finished yet
    if (trieIsEnd[node]) return 1;
    return 0;
}

// the slow full check: put the token behind the beam text and look at all sites
__device__ bool fullCheck(AllData* data, BeamState* state, const char* cleanedBeamText, TokenInfo* token)
{
    BeamState stateAfterToken = *state;
    char cleanedToken[MAX_TOKEN_LENGTH];
    int tokenLength = token->length;
    if (tokenLength > MAX_TOKEN_LENGTH) tokenLength = MAX_TOKEN_LENGTH;
    int blankLastBeamChar = 0;

    for (int k = 0; k < tokenLength; k++) {
        char outChar;
        int blankPrevious;
        cleanOneChar(&stateAfterToken, token->tokenChars[k], &outChar, &blankPrevious);
        cleanedToken[k] = outChar;
        if (blankPrevious == 1) {
            if (k == 0) blankLastBeamChar = 1;
            else cleanedToken[k - 1] = ' ';
        }
    }

    JoinedText text;
    text.cleanedBeamText = cleanedBeamText;
    text.cleanedTokenText = cleanedToken;
    text.beamTextLength = state->textLength;
    text.totalLength = state->textLength + tokenLength;
    text.blankLastBeamChar = blankLastBeamChar;

    bool textEndsInLiteral = false;
    if (stateAfterToken.mode != 0) textEndsInLiteral = true;
    return checkAllSites(&text, false, textEndsInLiteral,
        data->memberTrieChildren, data->memberTrieIsEnd,
        data->typeTrieChildren, data->typeTrieIsEnd,
        data->caseTrieChildren, data->caseTrieIsEnd);
}

// is this token allowed after this beam text
__device__ bool isTokenAllowed(AllData* data, BeamState* state, const char* cleanedBeamText, TokenInfo* token)
{
    if (token->tokenClass == 1) return state->stopTokenAllowed != 0;   // stop token
    if (token->tokenClass == 2) return true;                           // special token, always ok
    if (noTriesExist(data)) return true;

    int quickAnswer = quickDecision(data, state, token);
    if (quickAnswer != 2) return quickAnswer == 1;
    return fullCheck(data, state, cleanedBeamText, token);
}

// ================== end of device part ==================


// kernel 1: one thread for every beam, cleans the text and works out how it ends
__global__ void prepareBeamsKernel(AllData data, const char* beamTexts, char* cleanedTexts, BeamState* beamStates, int numBeams)
{
    int beam = blockIdx.x * blockDim.x + threadIdx.x;
    if (beam >= numBeams) return;
    int textStart = data.textOffsets[beam];
    int textLength = data.textOffsets[beam + 1] - textStart;
    BeamState state;
    prepareBeam(&data, beamTexts + textStart, textLength, cleanedTexts + textStart, &state);
    beamStates[beam] = state;
}

// kernel 2: one thread for every token, it loops over all beams
// writes 1 or 0 for every beam+token and counts how many tokens are allowed for each beam
__global__ void judgeTokensKernel(AllData data, int scoresWidth, int numBeams, unsigned char* allowedFlags, int* allowedCounts)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    bool insideScores = tokenId < scoresWidth;
    bool isRealToken = insideScores && tokenId < data.vocabSize;   // ids after the vocab are never allowed
    TokenInfo token;
    if (isRealToken) token = getTokenInfo(&data, tokenId);

    for (int beam = 0; beam < numBeams; beam++) {
        bool allowed = false;
        if (isRealToken) {
            BeamState state = data.beamStates[beam];
            allowed = isTokenAllowed(&data, &state, data.cleanedTexts + data.textOffsets[beam], &token);
        }
        if (insideScores) {
            if (allowed) allowedFlags[beam * scoresWidth + tokenId] = 1;
            else allowedFlags[beam * scoresWidth + tokenId] = 0;
        }
        // count how many threads of this warp said yes, and add it once
        unsigned yesVotes = __ballot_sync(0xffffffffu, allowed);
        if ((threadIdx.x & 31) == 0 && yesVotes != 0) atomicAdd(&allowedCounts[beam], __popc(yesVotes));
    }
}

// kernel 3: put -inf on every token that is not allowed
// but if a beam has no allowed token at all we leave it alone
// (float version)
__global__ void applyMaskFloat(float* scores, const unsigned char* allowedFlags, const int* allowedCounts, int scoresWidth, int rowStride, int numBeams)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    int beam = blockIdx.y;
    if (beam >= numBeams || tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = -INFINITY;
    }
}

// (half version)
__global__ void applyMaskHalf(__half* scores, const unsigned char* allowedFlags, const int* allowedCounts, int scoresWidth, int rowStride, int numBeams)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    int beam = blockIdx.y;
    if (beam >= numBeams || tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = __float2half(-INFINITY);
    }
}

// (bfloat16 version)
__global__ void applyMaskBf16(__nv_bfloat16* scores, const unsigned char* allowedFlags, const int* allowedCounts, int scoresWidth, int rowStride, int numBeams)
{
    int tokenId = blockIdx.x * blockDim.x + threadIdx.x;
    int beam = blockIdx.y;
    if (beam >= numBeams || tokenId >= scoresWidth) return;
    if (allowedCounts[beam] > 0 && allowedFlags[beam * scoresWidth + tokenId] == 0) {
        scores[(size_t)beam * rowStride + tokenId] = __float2bfloat16(-INFINITY);
    }
}

// the cpp file calls this when a VocabChecker is made or deleted
// this version of the code doesnt need it, its only here so the cpp file still links
void bump_vocab_generation()
{
}

// dtype_code: 0 = float32, 1 = float16, 2 = bfloat16
// (d_text_offsets_buf and text_offsets_cap are not used anymore, everything is in d_batch_text_buf)
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
    int max_text_per_beam)
{
    if (num_beams == 0 || scores_ptr == 0) return;

    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int scoresWidth = scores_vocab_size;

    // how long is the text of every beam (we only keep the last max_text_per_beam chars)
    vector<int> textOffsets(num_beams + 1, 0);
    vector<int> keepFrom(num_beams, 0);
    for (int beam = 0; beam < num_beams; beam++) {
        int fullLength = 0;
        int startAt = 0;
        if (beam < (int)batch_texts.size()) {
            fullLength = (int)batch_texts[beam].length();
            if (fullLength > max_text_per_beam) startAt = fullLength - max_text_per_beam;
        }
        keepFrom[beam] = startAt;
        textOffsets[beam + 1] = textOffsets[beam] + (fullLength - startAt);
    }
    int totalTextChars = textOffsets[num_beams];

    // where everything goes inside the one big gpu buffer (every part starts at a multiple of 16)
    int offsetsBytes = ((num_beams + 1) * 4 + 15) / 16 * 16;
    int textsStart = offsetsBytes;
    int cleanedTextsStart = (textsStart + totalTextChars + 15) / 16 * 16;
    int statesStart = (cleanedTextsStart + totalTextChars + 15) / 16 * 16;
    int countsStart = (statesStart + num_beams * (int)sizeof(BeamState) + 15) / 16 * 16;
    int flagsStart = (countsStart + num_beams * 4 + 15) / 16 * 16;
    int bytesNeeded = flagsStart + num_beams * scoresWidth + 16;

    // make the gpu buffer bigger if its too small
    if (*d_batch_text_buf == NULL || (size_t)bytesNeeded > *batch_text_cap) {
        if (*d_batch_text_buf != NULL) cudaFree(*d_batch_text_buf);
        *batch_text_cap = (size_t)bytesNeeded * 2;
        cudaMalloc((void**)d_batch_text_buf, *batch_text_cap);
    }
    char* gpuBase = *d_batch_text_buf;

    // pinned cpu memory so we can send the offsets and the texts with one copy
    static char* pinnedBuffer = NULL;
    static int pinnedBufferSize = 0;
    static cudaEvent_t copyFinished;
    static int eventCreated = 0;
    if (eventCreated == 0) {
        cudaEventCreateWithFlags(&copyFinished, cudaEventDisableTiming);
        eventCreated = 1;
    } else {
        cudaEventSynchronize(copyFinished);    // wait until the last copy is done reading the pinned buffer
    }
    int bytesToCopy = textsStart + totalTextChars;
    if (pinnedBuffer == NULL || bytesToCopy > pinnedBufferSize) {
        if (pinnedBuffer != NULL) cudaFreeHost(pinnedBuffer);
        pinnedBufferSize = bytesToCopy * 2 + 4096;
        cudaMallocHost((void**)&pinnedBuffer, pinnedBufferSize);
    }
    memcpy(pinnedBuffer, textOffsets.data(), (num_beams + 1) * sizeof(int));
    for (int beam = 0; beam < num_beams; beam++) {
        int length = textOffsets[beam + 1] - textOffsets[beam];
        if (length > 0) memcpy(pinnedBuffer + textsStart + textOffsets[beam], batch_texts[beam].data() + keepFrom[beam], length);
    }
    cudaMemcpyAsync(gpuBase, pinnedBuffer, bytesToCopy, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(copyFinished, stream);

    int* gpuAllowedCounts = (int*)(gpuBase + countsStart);
    cudaMemsetAsync(gpuAllowedCounts, 0, num_beams * sizeof(int), stream);

    AllData data;
    data.vocabOffsets = d_vocab_offsets;
    data.vocabChars = d_vocab_chars;
    data.isSpecialToken = d_is_special;
    data.isStopToken = d_is_stop;
    data.cleanedTexts = gpuBase + cleanedTextsStart;
    data.textOffsets = (const int*)gpuBase;
    data.beamStates = (const BeamState*)(gpuBase + statesStart);
    data.vocabSize = vocab_size;
    data.memberTrieChildren = d_trie_mem_children;
    data.memberTrieIsEnd = d_trie_mem_is_end;
    data.typeTrieChildren = d_trie_type_children;
    data.typeTrieIsEnd = d_trie_type_is_end;
    data.caseTrieChildren = d_trie_case_children;
    data.caseTrieIsEnd = d_trie_case_is_end;

    // step 1: clean the beam texts
    prepareBeamsKernel<<<(num_beams + 63) / 64, 64, 0, stream>>>(
        data, gpuBase + textsStart, gpuBase + cleanedTextsStart, (BeamState*)(gpuBase + statesStart), num_beams);

    // step 2: check all the tokens
    unsigned char* gpuAllowedFlags = (unsigned char*)(gpuBase + flagsStart);
    judgeTokensKernel<<<(scoresWidth + 255) / 256, 256, 0, stream>>>(
        data, scoresWidth, num_beams, gpuAllowedFlags, gpuAllowedCounts);

    // step 3: put -inf on the tokens that are not allowed
    dim3 threads(256, 1);
    dim3 blocks((scoresWidth + 255) / 256, num_beams);
    if (dtype_code == 0) {
        applyMaskFloat<<<blocks, threads, 0, stream>>>(
            (float*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, scoresWidth, row_stride, num_beams);
    }
    else if (dtype_code == 1) {
        applyMaskHalf<<<blocks, threads, 0, stream>>>(
            (__half*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, scoresWidth, row_stride, num_beams);
    }
    else {
        applyMaskBf16<<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)scores_ptr, gpuAllowedFlags, gpuAllowedCounts, scoresWidth, row_stride, num_beams);
    }
}