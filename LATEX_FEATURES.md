# LaTeX Editing Features in Neovim

Your neovim is now configured with Texifier-like autocompletion for LaTeX documents.

## Installed Plugins

1. **vimtex** - Core LaTeX plugin
2. **luasnip-latex-snippets.nvim** - Pre-built LaTeX snippets
3. **nvim-cmp with cmp-omni** - Command, label, and citation autocompletion
4. **nvim-autopairs** - Automatic bracket/brace insertion (LaTeX-enhanced)
5. **nvim-treesitter** - LaTeX syntax highlighting

## Features

### Auto-completion (nvim-cmp)
- **LaTeX commands**: Type `\` and see command suggestions
- **Citations**: Type `\cite{` to get citation completions
- **References**: Type `\ref{` to see label completions
- **Packages**: Type `\usepackage{` for package suggestions
- **Environments**: Type `\begin{` for environment completions

**Keybindings:**
- `Tab` - Accept completion
- `Ctrl+n` / `Ctrl+p` - Navigate up/down
- `Ctrl+Space` - Trigger completion manually
- `Ctrl+e` - Cancel completion

### Snippets (LuaSnip)
Common snippets that expand automatically:
- `beg` → Full `\begin{env}...\end{env}` block
- `fig` → Figure environment
- `tab` → Table environment  
- `item` → Itemize environment
- `enum` → Enumerate environment
- `eq` → Equation environment
- `ali` → Align environment
- `frac` → Fraction
- `sum` → Summation
- `int` → Integral
- And many more math symbols and environments!

**Navigation:**
- `Tab` - Jump to next snippet field
- `Shift+Tab` - Jump to previous snippet field

### Auto-pairs
**Automatic closing:**
- `$` → `$$` (inline math)
- `\\[` → `\\[\\]` (display math)
- `\\(` → `\\(\\)` (inline math alternative)
- `{` → `{}` (braces)
- `[` → `[]` (brackets)
- `(` → `()` (parentheses)

**Smart features:**
- Auto-closes when you complete environments via nvim-cmp
- Automatically closes braces in LaTeX commands

### VimTeX Commands
- `\\ll` - Toggle continuous compilation
- `\\lv` - View PDF
- `\\lc` - Clean auxiliary files
- `\\le` - View compilation errors
- `\\lt` - Table of contents
- `\\lk` - Stop compilation
- `\\lK` - Stop all compilations
- `\\li` - Show info about current document
- `\\lm` - Show imaps (insert mode mappings)

### Syntax Concealment
Mathematical symbols and Greek letters are displayed nicely in the editor:
- `\alpha` displays as α
- `\sum` displays as Σ
- `\int` displays as ∫
- Fractions, superscripts, and subscripts are formatted

## Configuration Files Modified

- `~/.config/nvim/lua/plugins/core.lua` - Added LaTeX plugins and configuration

## Testing

Two test files have been created:
1. `test_vimtex.tex` - Basic test
2. `latex_features_test.tex` - Comprehensive test with environments

Open either file in neovim to try the features:
```bash
nvim latex_features_test.tex
```

## PDF Viewer

Configured to use **Skim** (macOS). If you prefer another viewer, edit:
```lua
vim.g.vimtex_view_method = 'skim'  -- Change to 'zathura', 'preview', etc.
```

## Tips

1. **Start compilation**: Open a `.tex` file and press `\\ll`
2. **Use snippets**: Type the trigger word and expand with Tab
3. **LaTeX commands**: Type `\` and wait for completions to appear
4. **Math mode**: Type `$` and it auto-closes
5. **Environments**: Type `\begin{` and select from suggestions

Enjoy your Texifier-like LaTeX editing experience!
