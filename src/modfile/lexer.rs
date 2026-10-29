// SPDX-License-Identifier: GPL-3.0-or-later
//! Tokenizer for macro-expanded .mod source (Dynare's `DynareFlex.ll` subset).

use super::macro_proc::SrcLine;
use anyhow::{Result, bail};
use std::fmt;
use std::sync::Arc;

/// Source location `file:line`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loc {
    pub file: Arc<str>,
    pub line: usize,
}

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Num(f64),
    /// Quoted string (single or double quotes), without the quotes.
    Str(String),
    /// `$...$` TeX name in declarations (ignored).
    Tex(String),
    Punct(&'static str),
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    /// Source text of the token (used to rebuild unquoted option values such as paths).
    pub raw: String,
    pub loc: Loc,
    /// 1-based first and last column of the token on its line (0 when synthesized); used for
    /// the source locations of the preprocessor warnings.
    pub col: (usize, usize),
}

impl Token {
    pub fn is(&self, p: &str) -> bool {
        matches!(&self.tok, Tok::Punct(q) if *q == p)
    }
    pub fn is_ident(&self, s: &str) -> bool {
        matches!(&self.tok, Tok::Ident(i) if i == s)
    }
    pub fn ident(&self) -> Option<&str> {
        match &self.tok {
            Tok::Ident(i) => Some(i),
            _ => None,
        }
    }
}

/// `~` and `~=` only occur in MATLAB statements (`if ~isoctave`, see `parser_x`).
const PUNCT: [&str; 28] = [
    "==", "!=", "~=", "<=", ">=", "&&", "||", "+", "-", "*", "/", "^", "=", "(", ")", "[", "]", "{", "}", ",", ";",
    ":", "#", "<", ">", "!", "~", ".",
];

/// Tokenizes the macro processor output. Comments (`//`, `%`, `/* */`) are removed here, as
/// in `DynareFlex.ll` (the macro processor leaves them). A `verbatim; ... end;` block becomes
/// the three tokens `verbatim`, its MATLAB source as a string (comments included), `;` (see
/// `super::verbatim`).
pub fn tokenize(lines: &[SrcLine]) -> Result<Vec<Token>> {
    let mut out = Vec::new();
    let mut in_verbatim = false;
    let mut verbatim_src = String::new();
    // where the `/* */` comment open at the end of the previous line started
    let mut comment_start: Option<Loc> = None;
    for l in lines {
        let loc = Loc { file: l.file.clone(), line: l.line };
        let t = l.text.trim();
        if in_verbatim {
            if t.starts_with("end") && t[3..].trim_start().starts_with(';') {
                in_verbatim = false;
                let src = std::mem::take(&mut verbatim_src);
                out.push(Token { tok: Tok::Str(src.clone()), raw: src, loc: loc.clone(), col: (0, 0) });
                out.push(Token { tok: Tok::Punct(";"), raw: ";".into(), loc, col: (0, 0) });
            } else {
                verbatim_src.push_str(&l.text);
                verbatim_src.push('\n');
            }
            continue;
        }
        let mut in_comment = comment_start.is_some();
        let text = strip_comments(&l.text, &mut in_comment);
        match (in_comment, &comment_start) {
            (false, _) => comment_start = None,
            (true, None) => comment_start = Some(loc.clone()),
            (true, Some(_)) => {}
        }
        let t = text.trim();
        if t.starts_with("verbatim") && t[8..].trim_start().starts_with(';') {
            in_verbatim = true;
            out.push(Token { tok: Tok::Ident("verbatim".into()), raw: "verbatim".into(), loc, col: (0, 0) });
            continue;
        }
        lex_line(&text, &loc, &mut out)?;
    }
    if let Some(start) = comment_start {
        bail!("{start}: unterminated comment: `/*` without a closing `*/`");
    }
    if in_verbatim {
        bail!("unterminated verbatim block");
    }
    Ok(out)
}

/// Removes the `//`, `%` and `/* */` comments of one line, keeping quoted strings intact (a
/// closed `/* */` comment becomes a blank); `in_comment` carries an open `/* */` comment from
/// one line to the next.
fn strip_comments(s: &str, in_comment: &mut bool) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if *in_comment {
            if c == '*' && it.peek() == Some(&'/') {
                it.next();
                *in_comment = false;
                out.push(' ');
            }
            continue;
        }
        match c {
            '/' if it.peek() == Some(&'/') => break,
            '%' => break,
            '/' if it.peek() == Some(&'*') => {
                it.next();
                *in_comment = true;
            }
            '\'' | '"' => {
                out.push(c);
                for d in it.by_ref() {
                    out.push(d);
                    if d == c {
                        break;
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

fn lex_line(s: &str, loc: &Loc, out: &mut Vec<Token>) -> Result<()> {
    let b = s.as_bytes();
    let mut i = 0;
    let push = |out: &mut Vec<Token>, tok: Tok, raw: &str, st: usize, en: usize| {
        out.push(Token { tok, raw: raw.to_string(), loc: loc.clone(), col: (st + 1, en) })
    };
    'outer: while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' {
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E' || b[i] == b'd' || b[i] == b'D') {
                let mut k = i + 1;
                if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    i = k;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let raw = &s[st..i];
            let v: f64 = match raw.replace(['d', 'D'], "e").parse() {
                Ok(v) => v,
                Err(_) => bail!("{loc}: invalid number '{raw}'"),
            };
            push(out, Tok::Num(v), raw, st, i);
            continue;
        }
        // ---- the perpendicular symbol of complementarity conditions
        if s[i..].starts_with("_|_") || s[i..].starts_with('⟂') {
            let n = if s[i..].starts_with('⟂') { '⟂'.len_utf8() } else { 3 };
            push(out, Tok::Punct("⟂"), if n == 3 { "_|_" } else { "⟂" }, i, i + n);
            i += n;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            // on-the-fly declaration in a model block: `c|e` (endogenous), `e|x` (exogenous),
            // `beta|p` (parameter); the identifier keeps the suffix in its raw text
            let typed = b.get(i) == Some(&b'|')
                && b.get(i + 1).is_some_and(|t| b"eExXpP".contains(t))
                && !b.get(i + 2).is_some_and(|t| t.is_ascii_alphanumeric() || *t == b'_');
            let en = if typed { i + 2 } else { i };
            push(out, Tok::Ident(s[st..i].to_string()), &s[st..en], st, en);
            i = en;
            continue;
        }
        if c == '\'' || c == '"' || c == '$' {
            let Some(end) = s[i + 1..].find(c) else { bail!("{loc}: unterminated string") };
            let body = &s[i + 1..i + 1 + end];
            let tok = if c == '$' { Tok::Tex(body.to_string()) } else { Tok::Str(body.to_string()) };
            push(out, tok, &s[i..i + end + 2], i, i + end + 2);
            i += end + 2;
            continue;
        }
        for p in PUNCT {
            if s[i..].starts_with(p) {
                push(out, Tok::Punct(p), p, i, i + p.len());
                i += p.len();
                continue 'outer;
            }
        }
        bail!("{loc}: unexpected character '{c}'");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(s: &str) -> Vec<Tok> {
        let lines = vec![SrcLine { file: Arc::from("t.mod"), line: 1, text: s.to_string() }];
        tokenize(&lines).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn tokens() {
        assert_eq!(
            lex("x(-1) = 1.5e-3*.5 + y(+4) >= 'a/b';"),
            vec![
                Tok::Ident("x".into()),
                Tok::Punct("("),
                Tok::Punct("-"),
                Tok::Num(1.0),
                Tok::Punct(")"),
                Tok::Punct("="),
                Tok::Num(1.5e-3),
                Tok::Punct("*"),
                Tok::Num(0.5),
                Tok::Punct("+"),
                Tok::Ident("y".into()),
                Tok::Punct("("),
                Tok::Punct("+"),
                Tok::Num(4.0),
                Tok::Punct(")"),
                Tok::Punct(">="),
                Tok::Str("a/b".into()),
                Tok::Punct(";"),
            ]
        );
        assert_eq!(lex("x $\\alpha$ 2."), vec![Tok::Ident("x".into()), Tok::Tex("\\alpha".into()), Tok::Num(2.0)]);
    }

    #[test]
    fn comments() {
        let lines: Vec<SrcLine> = ["a = 1; // c", "/* x", "y */ b = 'x//y'; % tail", "verbatim;", "% m", "end;", "/*"]
            .iter()
            .enumerate()
            .map(|(i, t)| SrcLine { file: Arc::from("t.mod"), line: i + 1, text: t.to_string() })
            .collect();
        let e = tokenize(&lines).unwrap_err().to_string();
        assert!(e.contains("t.mod:7: unterminated comment"), "{e}");
        let t = tokenize(&lines[..6]).unwrap();
        let toks: Vec<Tok> = t.into_iter().map(|t| t.tok).collect();
        assert_eq!(
            toks,
            vec![
                Tok::Ident("a".into()),
                Tok::Punct("="),
                Tok::Num(1.0),
                Tok::Punct(";"),
                Tok::Ident("b".into()),
                Tok::Punct("="),
                Tok::Str("x//y".into()),
                Tok::Punct(";"),
                Tok::Ident("verbatim".into()),
                Tok::Str(
                    "% m
"
                    .into()
                ),
                Tok::Punct(";"),
            ]
        );
    }

    #[test]
    fn verbatim_and_errors() {
        let lines: Vec<SrcLine> = ["a;", "verbatim;", "x = y';", "end;", "b;"]
            .iter()
            .enumerate()
            .map(|(i, t)| SrcLine { file: Arc::from("t.mod"), line: i + 1, text: t.to_string() })
            .collect();
        let t = tokenize(&lines).unwrap();
        // a ; verbatim 'x = y'';\n' ; b ;
        assert_eq!(t.len(), 7);
        assert!(t[2].is_ident("verbatim"));
        assert!(matches!(&t[3].tok, Tok::Str(s) if s == "x = y';\n"));
        assert_eq!(t[5].loc.line, 5);
        let lines = vec![SrcLine { file: Arc::from("m.mod"), line: 7, text: "x = ?;".into() }];
        let e = tokenize(&lines).unwrap_err().to_string();
        assert!(e.contains("m.mod:7"), "{e}");
    }
}
