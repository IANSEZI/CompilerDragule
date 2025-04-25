import sys
import os
import pickle
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from tkinter import Tk, filedialog, messagebox, simpledialog, Text, Scrollbar, Frame, Button, Toplevel
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# ---------- Lexical Analyzer Section ----------
class TokenType(Enum):
    # Keywords
    VAR = auto()
    PRINT = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FUNCTION = auto()
    RETURN = auto()

    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()

    # Comparisons
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # Special
    EOF = auto()

class Token:
    def __init__(self, token_type, value=None, line=0, column=0):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        if self.value is not None:
            return f"Token({self.type}, {self.value}, pos={self.line}:{self.column})"
        return f"Token({self.type}, pos={self.line}:{self.column})"

class LexerError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if text else None
        self.line = 1
        self.column = 1
        self.keywords = {
            'var': TokenType.VAR,
            'print': TokenType.PRINT,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'function': TokenType.FUNCTION,
            'return': TokenType.RETURN,
        }

    def error(self, message):
        raise LexerError(f"{message} at line {self.line}, column {self.column}")

    def advance(self):
        if self.current_char == '\n':
            self.line += 1
            self.column = 0
        self.pos += 1
        self.column += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            self.advance()
        elif self.current_char == '/' and self.peek() == '*':
            self.advance()
            self.advance()
            while self.current_char is not None:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()

    def number(self):
        result = ''
        line = self.line
        column = self.column
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        if self.current_char == '.' and self.peek() and self.peek().isdigit():
            result += self.current_char
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(TokenType.NUMBER, float(result), line, column)
        return Token(TokenType.NUMBER, int(result), line, column)

    def string(self):
        result = ''
        line = self.line
        column = self.column
        self.advance()  # Skip opening quote
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\' and self.peek():
                self.advance()
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}
                result += escape_map.get(self.current_char, self.current_char)
            else:
                result += self.current_char
            self.advance()
        if self.current_char is None:
            raise LexerError(f"Unterminated string literal at line {line}, column {column}")
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, result, line, column)

    def identifier(self):
        result = ''
        line = self.line
        column = self.column
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        token_type = self.keywords.get(result, TokenType.IDENTIFIER)
        return Token(token_type, result, line, column)

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char == '/' and (self.peek() == '/' or self.peek() == '*'):
                self.skip_comment()
                continue
            if self.current_char.isdigit():
                return self.number()
            if self.current_char == '"':
                return self.string()
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()

            line = self.line
            column = self.column
            char = self.current_char
            self.advance()

            if char == '+': return Token(TokenType.PLUS, '+', line, column)
            if char == '-': return Token(TokenType.MINUS, '-', line, column)
            if char == '*': return Token(TokenType.MULTIPLY, '*', line, column)
            if char == '/': return Token(TokenType.DIVIDE, '/', line, column)
            if char == '=':
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.EQUAL, '==', line, column)
                return Token(TokenType.ASSIGN, '=', line, column)
            if char == '!':
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.NOT_EQUAL, '!=', line, column)
                self.error("Expected '=' after '!'")
            if char == '<':
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.LESS_EQUAL, '<=', line, column)
                return Token(TokenType.LESS_THAN, '<', line, column)
            if char == '>':
                if self.current_char == '=':
                    self.advance()
                    return Token(TokenType.GREATER_EQUAL, '>=', line, column)
                return Token(TokenType.GREATER_THAN, '>', line, column)
            if char == '(': return Token(TokenType.LPAREN, '(', line, column)
            if char == ')': return Token(TokenType.RPAREN, ')', line, column)
            if char == '{': return Token(TokenType.LBRACE, '{', line, column)
            if char == '}': return Token(TokenType.RBRACE, '}', line, column)
            if char == '[': return Token(TokenType.LBRACKET, '[', line, column)
            if char == ']': return Token(TokenType.RBRACKET, ']', line, column)
            if char == ';': return Token(TokenType.SEMICOLON, ';', line, column)
            if char == ',': return Token(TokenType.COMMA, ',', line, column)

            self.error(f"Unrecognized character '{char}'")

        return Token(TokenType.EOF, None, self.line, self.column)

    def tokenize(self):
        tokens = []
        while True:
            try:
                token = self.get_next_token()
                tokens.append(token)
                if token.type == TokenType.EOF:
                    break
            except LexerError as e:
                print(f"Lexer Error: {e}")
                self.advance()
        return tokens

# ---------- Grammar Handling Section ----------
class Grammar:
    def __init__(self):
        self.productions = OrderedDict()
        self.non_terminals = set()
        self.terminals = set()
        self.start_symbol = None
        self.valid = False

    def validate_grammar(self):
        if not self.start_symbol:
            raise ValueError("No start symbol defined")
        if self.start_symbol not in self.non_terminals:
            raise ValueError("Start symbol not in non-terminals")
        self.valid = True

    def load_from_file(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if '->' not in line:
                        raise SyntaxError(f"Invalid production format at line {line_num}")

                    lhs, rhs = line.split('->', 1)
                    lhs = lhs.strip()
                    rhs = [prod.strip().split() for prod in rhs.split('|')]

                    if not lhs:
                        raise SyntaxError(f"Empty LHS at line {line_num}")

                    if lhs not in self.productions:
                        self.productions[lhs] = []

                    for prod in rhs:
                        if not prod:
                            raise SyntaxError(f"Empty production at line {line_num}")
                        self.productions[lhs].append(prod)

                        for symbol in prod:
                            if symbol == 'ε':
                                continue
                            if symbol.isupper():
                                self.non_terminals.add(symbol)
                            else:
                                self.terminals.add(symbol)

                    if not self.start_symbol:
                        self.start_symbol = lhs

            self.terminals.add('$')
            self.validate_grammar()
        except Exception as e:
            self.valid = False
            raise RuntimeError(f"Grammar loading failed: {str(e)}")

# ---------- LL(1) Parser Section ----------
class LL1Parser:
    class ParseTreeNode:
        def __init__(self, symbol, children=None, token=None):
            self.symbol = symbol
            self.children = children or []
            self.token = token

        def __repr__(self, level=0):
            ret = "  " * level + f"{self.symbol}"
            if self.token:
                ret += f" ({self.token.value})"
            ret += "\n"
            for child in self.children:
                ret += child.__repr__(level+1)
            return ret

    def __init__(self, grammar):
        self.grammar = grammar
        self.first = defaultdict(set)
        self.follow = defaultdict(set)
        self.table = defaultdict(dict)
        self.parse_tree = None
        self.build()

    def compute_first(self):
        for nt in self.grammar.non_terminals:
            self.first[nt] = set()

        changed = True
        while changed:
            changed = False
            for nt, productions in self.grammar.productions.items():
                for prod in productions:
                    for symbol in prod:
                        if symbol in self.grammar.terminals:
                            before = len(self.first[nt])
                            self.first[nt].add(symbol)
                            if len(self.first[nt]) > before:
                                changed = True
                            break
                        elif symbol in self.grammar.non_terminals:
                            before = len(self.first[nt])
                            self.first[nt].update(self.first[symbol] - {'ε'})
                            if 'ε' not in self.first[symbol]:
                                if len(self.first[nt]) > before:
                                    changed = True
                                break
                    else:
                        before = len(self.first[nt])
                        self.first[nt].add('ε')
                        if len(self.first[nt]) > before:
                            changed = True

    def compute_follow(self):
        self.follow[self.grammar.start_symbol].add('$')
        changed = True
        while changed:
            changed = False
            for nt, productions in self.grammar.productions.items():
                for prod in productions:
                    for i, symbol in enumerate(prod):
                        if symbol in self.grammar.non_terminals:
                            before = len(self.follow[symbol])
                            next_symbols = prod[i+1:]
                            first_of_next = self.get_first(next_symbols)
                            self.follow[symbol].update(first_of_next - {'ε'})
                            if 'ε' in first_of_next or i == len(prod)-1:
                                self.follow[symbol].update(self.follow[nt])
                            if len(self.follow[symbol]) > before:
                                changed = True

    def get_first(self, symbols):
        if not symbols:
            return {'ε'}
        first = set()
        for symbol in symbols:
            if symbol in self.grammar.terminals:
                first.add(symbol)
                break
            elif symbol in self.grammar.non_terminals:
                first.update(self.first[symbol] - {'ε'})
                if 'ε' not in self.first[symbol]:
                    break
            if symbol == 'ε':
                first.add('ε')
        else:
            first.add('ε')
        return first

    def build_table(self):
        for nt, productions in self.grammar.productions.items():
            for prod in productions:
                first = self.get_first(prod)
                for terminal in first - {'ε'}:
                    self.table[nt][terminal] = prod
                if 'ε' in first:
                    for terminal in self.follow[nt]:
                        self.table[nt][terminal] = prod

    def build(self):
        self.compute_first()
        self.compute_follow()
        self.build_table()

    def parse(self, tokens):
        stack = [self.grammar.start_symbol]
        self.parse_tree = self.ParseTreeNode(self.grammar.start_symbol)
        node_stack = [self.parse_tree]
        token_iter = iter(tokens)
        current_token = next(token_iter, None)
        
        while stack:
            top = stack.pop()
            current_node = node_stack.pop()
            
            if top in self.grammar.terminals:
                if current_token and current_token.type.name == top:
                    current_node.token = current_token
                    current_token = next(token_iter, None)
                else:
                    raise SyntaxError(f"Expected {top}, got {current_token.type if current_token else 'EOF'}")
            elif top in self.grammar.non_terminals:
                if current_token:
                    if current_token.type.name in self.table[top]:
                        production = self.table[top][current_token.type.name]
                    elif 'ε' in self.table[top]:
                        production = ['ε']
                    else:
                        raise SyntaxError(f"No production for {top} on {current_token.type}")
                else:
                    if '$' in self.table[top]:
                        production = self.table[top]['$']
                    else:
                        raise SyntaxError(f"No production for {top} on EOF")
                
                if production != ['ε']:
                    for symbol in reversed(production):
                        stack.append(symbol)
                        new_node = self.ParseTreeNode(symbol)
                        current_node.children.append(new_node)
                        node_stack.append(new_node)
            else:
                raise SyntaxError(f"Unknown symbol {top}")
        
        return True
# ========== LR(0) Parser ==========
class LR0Parser:
    class ParseTreeNode:
        def __init__(self, symbol, children=None, token=None):
            self.symbol = symbol
            self.children = children or []
            self.token = token

        def __repr__(self, level=0):
            ret = "  " * level + f"{self.symbol}"
            if self.token:
                ret += f" ({self.token.value})"
            ret += "\n"
            for child in self.children:
                ret += child.__repr__(level+1)
            return ret

    def __init__(self, grammar):
        self.grammar = grammar
        self.states = []
        self.transitions = {}
        self.action_table = {}
        self.goto_table = {}
        self.parse_tree = None
        self.build()

    def closure(self, items):
        closure = set(items)
        changed = True
        while changed:
            changed = False
            for nt, prod, pos in list(closure):
                if pos < len(prod) and prod[pos] in self.grammar.non_terminals:
                    for production in self.grammar.productions[prod[pos]]:
                        item = (prod[pos], tuple(production), 0)
                        if item not in closure:
                            closure.add(item)
                            changed = True
        return frozenset(closure)

    def goto(self, items, symbol):
        goto = set()
        for nt, prod, pos in items:
            if pos < len(prod) and prod[pos] == symbol:
                goto.add((nt, prod, pos+1))
        return self.closure(goto)

    def build(self):
        initial_item = (self.grammar.start_symbol + "'", (self.grammar.start_symbol,), 0)
        start_state = self.closure({initial_item})
        self.states.append(start_state)
        queue = [start_state]
        state_ids = {start_state: 0}
        
        while queue:
            current = queue.pop(0)
            symbols = set()
            for item in current:
                nt, prod, pos = item
                if pos < len(prod):
                    symbols.add(prod[pos])
            
            for symbol in symbols:
                new_state = self.goto(current, symbol)
                if new_state not in state_ids:
                    state_ids[new_state] = len(self.states)
                    self.states.append(new_state)
                    queue.append(new_state)
                self.transitions[(state_ids[current], symbol)] = state_ids[new_state]
        
        # Build ACTION and GOTO tables
        for i, state in enumerate(self.states):
            self.action_table[i] = {}
            self.goto_table[i] = {}
            
            for item in state:
                nt, prod, pos = item
                if pos == len(prod):
                    if nt == self.grammar.start_symbol + "'":
                        self.action_table[i]['$'] = ('accept',)
                    else:
                        for t in self.grammar.terminals.union({'$'}):
                            if t in self.grammar.follow.get(nt, set()):
                                self.action_table[i][t] = ('reduce', nt, prod)
                else:
                    next_sym = prod[pos]
                    if next_sym in self.grammar.terminals:
                        if (i, next_sym) in self.transitions:
                            self.action_table[i][next_sym] = ('shift', self.transitions[(i, next_sym)])
                    elif next_sym in self.grammar.non_terminals:
                        if (i, next_sym) in self.transitions:
                            self.goto_table[i][next_sym] = self.transitions[(i, next_sym)]
            
            for t in self.grammar.terminals:
                if t not in self.action_table[i]:
                    self.action_table[i][t] = ('error',)
            
            self.action_table[i]['$'] = self.action_table[i].get('$', ('error',))

    def parse(self, tokens):
        stack = [0]
        symbol_stack = []
        node_stack = []
        token_iter = iter(tokens)
        current_token = next(token_iter, Token(TokenType.EOF, None))
        
        while True:
            state = stack[-1]
            action = self.action_table[state].get(
                current_token.type.name if current_token else '$', 
                ('error',)
            )
            
            if action[0] == 'shift':
                stack.append(action[1])
                symbol_stack.append(current_token.type.name)
                node_stack.append(self.ParseTreeNode(current_token.type.name, token=current_token))
                current_token = next(token_iter, Token(TokenType.EOF, None))
            elif action[0] == 'reduce':
                nt, prod = action[1], action[2]
                node = self.ParseTreeNode(nt)
                children = []
                for _ in range(len(prod)):
                    stack.pop()
                    symbol_stack.pop()
                    children.insert(0, node_stack.pop())
                node.children = children
                goto_state = self.goto_table[stack[-1]][nt]
                stack.append(goto_state)
                symbol_stack.append(nt)
                node_stack.append(node)
            elif action[0] == 'accept':
                self.parse_tree = node_stack[-1]
                return True
            else:
                raise SyntaxError(f"Syntax error at {current_token}")

# ---------- File Dialog Helpers ----------
class FileDialogHelper:
    def __init__(self):
        self.root = Tk()
        self.root.withdraw()

    def select_file(self, title="Select File", filetypes=(("Text files", "*.txt"), ("All files", "*.*"))):
        return filedialog.askopenfilename(title=title, filetypes=filetypes)

    def select_save_location(self, default_name="report.pdf"):
        return filedialog.asksaveasfilename(
            title="Save Report As",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))
        )

# ---------- Report Generator ----------
class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.elements = []
        self.errors = []
    
    def add_heading(self, text, level=1):
        self.elements.append(Paragraph(f"<font size={12+level*2}><b>{text}</b></font>", self.styles["Normal"]))
    
    def add_paragraph(self, text):
        self.elements.append(Paragraph(text, self.styles["Normal"]))
    
    def add_table(self, data, headers):
        table_data = [headers] + data
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.elements.append(table)
    
    def add_parse_tree(self, root_node):
        tree_data = []
        def traverse(node, level=0):
            prefix = "  " * level
            label = f"{node.symbol}"
            if node.token:
                label += f" ({node.token.value})"
            tree_data.append([prefix + label])
            for child in node.children:
                traverse(child, level+1)
        
        traverse(root_node)
        self.add_table(tree_data, ["Parse Tree Structure"])
    
    def generate(self, filename):
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            doc.build(self.elements)
            return True
        except Exception as e:
            self.errors.append(f"PDF Generation Error: {str(e)}")
            return False

# ---------- Main Application Class ----------
class CompilerAnalyzerApp:
    def __init__(self):
        self.file_helper = FileDialogHelper()
        self.current_grammar = None
        self.current_tokens = []
        self.current_parse_tree = None
        self.report = PDFReportGenerator()
        

    def save_tokens(self):
        """Save tokens to a file chosen by the user"""
        filename = self.file_helper.select_save_location(default_name="tokens.pkl")
        if not filename:
            return False
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.current_tokens, f)
            return True
        except Exception as e:
            self._handle_error(f"Failed to save tokens: {str(e)}")
            return False
        
    def load_tokens(self):
        """Load tokens from a file selected by the user"""
        filename = self.file_helper.select_file(title="Select Token File", filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
        if not filename:
            return False
        try:
            with open(filename, 'rb') as f:
                self.current_tokens = pickle.load(f)
            return True
        except Exception as e:
            self._handle_error(f"Failed to load tokens: {str(e)}")
            return False  
    
    def run_lexical_analysis(self):
        # [Previous implementation with added token saving]
        if self.run_lexical_analysis_impl():
            if self.save_tokens():
                return True
        return False

    def run_ll1_analysis(self):
        if not self._validate_analysis_prerequisites():
            return False
        
        # Ensure we have the latest tokens
        if not self.load_tokens():
            return False

        # [Rest of LL1 implementation remains the same]
        pass

    def run_lr0_analysis(self):
        if not self._validate_analysis_prerequisites():
            return False
        
        # Ensure we have the latest tokens
        if not self.load_tokens():
            return False

        try:
            parser = LR0Parser(self.current_grammar)
            success = parser.parse(self.current_tokens)
            
            if not success:
                if not self._handle_error("LR(0) parsing failed", fatal=False):
                    return False
            
            self.report.add_heading("LR(0) Parse Tree", level=2)
            self.report.add_parse_tree(parser.parse_tree)
            
            # Add state table
            state_data = []
            for i, state in enumerate(parser.states):
                state_data.append([f"State {i}", str(state)])
            self.report.add_table(state_data, ["State", "Items"])
            
            # Add action/goto tables
            action_data = []
            for state, actions in parser.action_table.items():
                for symbol, action in actions.items():
                    action_data.append([f"State {state}", symbol, str(action)])
            self.report.add_table(action_data, ["State", "Symbol", "Action"])
            
            goto_data = []
            for state, gotos in parser.goto_table.items():
                for symbol, goto_state in gotos.items():
                    goto_data.append([f"State {state}", symbol, f"State {goto_state}"])
            self.report.add_table(goto_data, ["State", "Non-Terminal", "Goto State"])
            
            messagebox.showinfo("Success", "LR(0) parsing completed successfully!")
            return True

        except Exception as e:
            if not self._handle_error(f"LR(0) Parsing Error: {str(e)}", fatal=False):
                return False


    def _create_code_input_dialog(self):
        """Create a user-friendly code input dialog"""
        dialog = Toplevel(self.file_helper.root)
        dialog.title("Enter Source Code")
        
        text_frame = Frame(dialog)
        text_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        text_area = Text(text_frame, width=80, height=20, wrap='word', font=('Courier', 10))
        text_area.pack(side='left', fill='both', expand=True)
        
        scrollbar = Scrollbar(text_frame, command=text_area.yview)
        scrollbar.pack(side='right', fill='y')
        text_area.config(yscrollcommand=scrollbar.set)
        
        # Add instructions
        text_area.insert('end', "/* Enter your source code here\n")
        text_area.insert('end', "   - Type or paste your code\n")
        text_area.insert('end', "   - Click 'Submit' when done */\n\n")
        
        button_frame = Frame(dialog)
        button_frame.pack(pady=5)
        
        self.user_code = None
        
        def submit():
            self.user_code = text_area.get('1.0', 'end-1c')
            dialog.destroy()
        
        submit_btn = Button(button_frame, text="Submit", command=submit)
        submit_btn.pack(side='left', padx=5)
        
        cancel_btn = Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_btn.pack(side='left', padx=5)
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        dialog.transient(self.file_helper.root)
        dialog.grab_set()
        dialog.wait_window()
        
        return self.user_code

    def _get_input_source(self):
        choice = messagebox.askquestion(
            "Input Source",
            "How would you like to provide the source code?\n\n"
            "• 'Yes' - Select a file\n"
            "• 'No' - Type/paste code manually",
            icon='question'
        )
        
        if choice == 'yes':
            filename = self.file_helper.select_file("Select Source Code File")
            if not filename:
                return None
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self._handle_error(f"Failed to read file: {str(e)}")
                return None
        else:
            return self._create_code_input_dialog()

    def run_lexical_analysis(self):
        while True:
            try:
                code = self._get_input_source()
                if not code:
                    if not self._handle_error("No source code provided", fatal=False):
                        return False
                    continue

                lexer = Lexer(code)
                self.current_tokens = lexer.tokenize()
                
                if not self.current_tokens or (
                    len(self.current_tokens) == 1 and 
                    self.current_tokens[0].type == TokenType.EOF
                ):
                    if not self._handle_error("No valid tokens generated from input", fatal=False):
                        return False
                    continue

                self.report.add_heading("Lexical Analysis Results", level=2)
                
                token_data = []
                for token in self.current_tokens:
                    if token.type == TokenType.EOF:
                        continue
                    token_info = [
                        str(token.type).split('.')[-1],
                        f"'{token.value}'" if token.value is not None else 'None',
                        f"Line {token.line}, Column {token.column}"
                    ]
                    token_data.append(token_info)
                
                self.report.add_table(token_data, ["Token Type", "Value", "Position"])
                
                token_count = len([t for t in self.current_tokens if t.type != TokenType.EOF])
                messagebox.showinfo(
                    "Success", 
                    f"Lexical analysis completed!\nFound {token_count} tokens."
                )
                return True

            except LexerError as e:
                if not self._handle_error(f"Lexical Error: {str(e)}", fatal=False):
                    return False
            except Exception as e:
                if not self._handle_error(f"Unexpected Error: {str(e)}", fatal=False):
                    return False

    def load_grammar(self):
        while True:
            try:
                filename = self.file_helper.select_file("Select Grammar File")
                if not filename:
                    if not self._handle_error("No grammar file selected", fatal=False):
                        return False
                    continue

                self.current_grammar = Grammar()
                self.current_grammar.load_from_file(filename)
                
                if not self.current_grammar.valid:
                    if not self._handle_error("Invalid grammar", fatal=False):
                        return False
                    continue

                self.report.add_heading("Grammar Specification", level=2)
                grammar_data = [
                    [nt, ' -> ', ' | '.join(' '.join(p) for p in prods)]
                    for nt, prods in self.current_grammar.productions.items()
                ]
                self.report.add_table(grammar_data, ["Non-Terminal", "", "Productions"])
                messagebox.showinfo("Success", "Grammar loaded successfully!")
                return True

            except Exception as e:
                if not self._handle_error(f"Grammar Error: {str(e)}", fatal=False):
                    return False

    def run_ll1_analysis(self):
        if not self._validate_analysis_prerequisites():
            return False

        while True:
            try:
                parser = LL1Parser(self.current_grammar)
                success = parser.parse(self.current_tokens)
                
                if not success:
                    if not self._handle_error("LL(1) parsing failed", fatal=False):
                        return False
                    continue

                self.report.add_heading("LL(1) Parse Tree", level=2)
                self.report.add_parse_tree(parser.parse_tree)
                
                ff_data = [
                    [nt, str(parser.first[nt]), str(parser.follow[nt])]
                    for nt in self.current_grammar.non_terminals
                ]
                self.report.add_table(ff_data, ["Non-Terminal", "First Set", "Follow Set"])
                
                messagebox.showinfo("Success", "LL(1) parsing completed successfully!")
                return True

            except Exception as e:
                if not self._handle_error(f"LL(1) Parsing Error: {str(e)}", fatal=False):
                    return False

    def generate_report(self):
        filename = self.file_helper.select_save_location()
        if not filename:
            return False
        
        if self.report.generate(filename):
            messagebox.showinfo(
                "Success",
                f"Report successfully generated at:\n{filename}"
            )
            return True
        else:
            self._handle_error("Failed to generate report:\n" + "\n".join(self.report.errors))
            return False

    def _handle_error(self, message, fatal=False):
        error_msg = f"ERROR: {message}"
        print(error_msg)
        self.report.errors.append(error_msg)
        
        if fatal:
            messagebox.showerror("Fatal Error", message)
            return False
        
        retry = messagebox.askretrycancel(
            "Error Occurred",
            f"{message}\n\nWould you like to try again?"
        )
        return retry

    def _validate_analysis_prerequisites(self):
        if not self.current_grammar or not self.current_grammar.valid:
            self._handle_error("No valid grammar loaded - please load grammar first")
            return False
        if not self.current_tokens:
            self._handle_error("No tokens available - please run lexical analysis first")
            return False
        return True

# ---------- Main Execution ----------
if __name__ == "__main__":
    app = CompilerAnalyzerApp()
    
    print("=== Compiler Analysis Tool ===")
    print("This tool will guide you through the compilation process\n")
    
    while True:
        try:
            print("\nMain Menu:")
            print("1. Perform Lexical Analysis")
            print("2. Load Grammar")
            print("3. Run LL(1) Parser")
            print("4. Run LR(0) Parser")
            print("5. Generate Report")
            print("6. Exit")
            
            choice = input("Please enter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\nStarting Lexical Analysis...")
                app.run_lexical_analysis()
            elif choice == '2':
                print("\nLoading Grammar...")
                app.load_grammar()
            elif choice == '3':
                print("\nRunning LL(1) Parser...")
                app.run_ll1_analysis()
            elif choice == '4':
                print("\nRunning LR(0) Parser...")
                app.run_lr0_analysis()  
            elif choice == '5':
                print("\nGenerating Report...")
                app.generate_report()
            elif choice == '6':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please enter a number between 1-5.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            if messagebox.askyesno("Exit", "Do you really want to exit?"):
                break
            continue
        except Exception as e:
            print(f"\nFatal Error: {str(e)}")
            if not messagebox.askretrycancel("Fatal Error", f"{str(e)}\n\nWould you like to continue?"):
                break

    # Cleanup
    app.file_helper.root.destroy()
    print("\nThank you for using the Compiler Analysis Tool!")