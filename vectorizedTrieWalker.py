import numpy as np

class VectorizedTrieWalker:
    def __init__(self, domains, vocab_identifiers):
        """
        domains: dict mapping a state name to valid string completions.
                 e.g., {"Math": ["max", "min", "abs", "PI"]}
        vocab_identifiers: list of the identifier portion of every token in the vocab.
                 e.g., ["", "x", "gic", "max", "in", "Of("]
        """
        self.V = len(vocab_identifiers)
        self.domains = domains
        
        # 1. Compile domains into a flat Transition Matrix
        self._build_transition_matrix()
        
        # 2. Compile tokens into a padded Character Matrix
        self._build_token_matrix(vocab_identifiers)

    def _build_transition_matrix(self):
        # Extract unique characters to create integer mappings
        chars = sorted({ch for words in self.domains.values() for word in words for ch in word})
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(chars)} # 0 is reserved for unknown chars
        
        # Initialize trie nodes
        self.children = [{}]
        self.terminal = [False]
        self.roots = {}
        
        # Build standard trie
        for domain_name, words in self.domains.items():
            root_id = len(self.children)
            self.children.append({})
            self.terminal.append(False)
            self.roots[domain_name] = root_id
            
            for word in words:
                node = root_id
                for ch in word:
                    if ch not in self.children[node]:
                        self.children[node][ch] = len(self.children)
                        self.children.append({})
                        self.terminal.append(False)
                    node = self.children[node][ch]
                self.terminal[node] = True

        self.DEAD = len(self.children)
        
        # Flatten trie into a 2D integer Transition Matrix: [Num_Nodes + 1, Num_Chars + 1]
        self.trans = np.full((self.DEAD + 1, len(chars) + 1), self.DEAD, dtype=np.int64)
        for node, edges in enumerate(self.children):
            for ch, nxt_node in edges.items():
                self.trans[node, self.char_to_id[ch]] = nxt_node
                
        self.is_terminal = np.array(self.terminal + [False], dtype=bool)
        self.is_alive = np.ones(self.DEAD + 1, dtype=bool)
        self.is_alive[self.DEAD] = False

    def _build_token_matrix(self, vocab_identifiers):
        # Find longest identifier length
        self.L = max((len(u) for u in vocab_identifiers), default=0)
        
        # Token character matrix: [V, L]
        self.uc = np.zeros((self.V, max(self.L, 1)), dtype=np.int64)
        self.u_len = np.zeros(self.V, dtype=np.int64)
        
        for i, ident in enumerate(vocab_identifiers):
            self.u_len[i] = len(ident)
            for j, ch in enumerate(ident):
                self.uc[i, j] = self.char_to_id.get(ch, 0) # Map to int ID

    def get_mask(self, receiver, base_name):
        """Returns a boolean array of length V indicating valid tokens."""
        # 1. Start at the root node for this domain (e.g., "Math")
        curr_node = self.roots.get(receiver, self.DEAD)
        
        # 2. Advance the state machine using the already-typed base_name
        for ch in base_name:
            if curr_node == self.DEAD: 
                break
            curr_node = self.trans[curr_node, self.char_to_id.get(ch, 0)]
            
        if curr_node == self.DEAD:
            return np.zeros(self.V, dtype=bool) # Nothing is valid
            
        # 3. Initialize the state array for all V tokens
        state = np.full(self.V, curr_node, dtype=np.int64)
        
        # 4. Execute the batched walk for the identifier characters
        for l in range(self.L):
            chars_at_l = self.uc[:, l]
            nxt_state = self.trans[state, chars_at_l]
            is_active = self.u_len > l
            state = np.where(is_active, nxt_state, state)
            
        # 5. Apply the structural rules (whitespace and punctuation)
        base_is_terminal = self.is_terminal[curr_node]
        ident_is_terminal = self.is_terminal[state]
        ident_is_alive = self.is_alive[state]
        
        # A token is valid if:
        # A. It has whitespace/no-ident AND the base_name is already a complete word.
        # B. It has leftover punctuation AND the combined word is a complete word.
        # C. It has no punctuation AND the combined word is a valid prefix (alive).
        mask = np.where(self.has_ws | (self.u_len == 0), base_is_terminal,
               np.where(self.has_rest, ident_is_terminal, ident_is_alive))
               
        return mask