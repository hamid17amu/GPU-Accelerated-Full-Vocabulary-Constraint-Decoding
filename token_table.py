import re

_WS = re.compile(r"^\s*")
_ID = re.compile(r"^[A-Za-z_$][\w$]*")

class TokenTable:
    def __init__(self, pieces):
        self.pieces = pieces
        self.has_ws = []
        self.ident = []
        self.rest = []
        self.is_special = []

        for p in pieces:
            if p is None:
                self.has_ws.append(False)
                self.ident.append("")
                self.rest.append("")
                self.is_special.append(True)
                continue
            
            # Decompose the token into its three structural parts
            w_match = _WS.match(p)
            w = w_match.group(0)
            p_no_ws = p[len(w):]
            
            id_match = _ID.match(p_no_ws)
            if id_match:
                ident = id_match.group(0)
                rest = p_no_ws[len(ident):]
            else:
                ident = ""
                rest = p_no_ws
                
            self.has_ws.append(bool(w))
            self.ident.append(ident)
            self.rest.append(rest)
            self.is_special.append(False)