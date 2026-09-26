"""The compile pipeline is split into four layers that can each be
exercised on their own, without running the whole compiler:

1.  the lexer (``TokenStream`` and newline normalization)
2.  the parser (token stream in, node tree out, no symbol registration)
3.  symbol analysis (``idtracking`` on a node tree)
4.  constant folding (``optimizer.optimize`` on a node tree)
"""

import pytest

from jinja2 import Environment
from jinja2 import nodes
from jinja2 import Template
from jinja2.bccache import FileSystemBytecodeCache
from jinja2.idtracking import symbols_for_node
from jinja2.lexer import get_lexer
from jinja2.lexer import Lexer
from jinja2.lexer import LexerConfig
from jinja2.lexer import normalize_newlines
from jinja2.lexer import Token
from jinja2.lexer import TokenStream
from jinja2.loaders import DictLoader
from jinja2.optimizer import optimize
from jinja2.parser import Parser


class TestLexerLayer:
    def test_token_sequence_without_environment(self):
        lexer = Lexer(LexerConfig())
        stream = lexer.tokenize("{% if a %}{{ b }}{% endif %}")

        assert [(t.type, t.value) for t in stream] == [
            ("block_begin", "{%"),
            ("name", "if"),
            ("name", "a"),
            ("block_end", "%}"),
            ("variable_begin", "{{"),
            ("name", "b"),
            ("variable_end", "}}"),
            ("block_begin", "{%"),
            ("name", "endif"),
            ("block_end", "%}"),
        ]

    def test_newline_normalization_without_environment(self):
        lexer = Lexer(LexerConfig(newline_sequence="\r\n", keep_trailing_newline=True))
        stream = lexer.tokenize("a\nb\r\nc\r\n")
        assert [(t.type, t.value) for t in stream] == [("data", "a\r\nb\r\nc\r\n")]

    def test_normalize_newlines(self):
        assert normalize_newlines("a\rb\r\nc\nd") == "a\nb\nc\nd"
        assert normalize_newlines("a\nb", "\r\n") == "a\r\nb"

    def test_config_from_environment(self):
        env = Environment(trim_blocks=True, newline_sequence="\r\n")
        config = LexerConfig.from_environment(env)
        assert config.trim_blocks is True
        assert config.newline_sequence == "\r\n"
        assert config.block_start_string == "{%"

    def test_config_equality_and_hash(self):
        assert LexerConfig(trim_blocks=True) == LexerConfig(trim_blocks=True)
        assert hash(LexerConfig()) == hash(LexerConfig())
        assert LexerConfig() != LexerConfig(lstrip_blocks=True)

    def test_lexer_cache_shared_for_equal_config(self):
        env_a = Environment(trim_blocks=True)
        env_b = Environment(trim_blocks=True)
        assert get_lexer(env_a) is get_lexer(env_b)
        assert get_lexer(env_a).config == LexerConfig(trim_blocks=True)

    def test_lexer_still_accepts_environment(self, env):
        lexer = Lexer(env)
        assert lexer.config == LexerConfig.from_environment(env)

    def test_token_stream_operations(self):
        stream = TokenStream(
            [
                Token(1, "name", "a"),
                Token(1, "block_end", "%}"),
            ],
            "foo",
            "bar",
        )
        assert stream.current.type == "name"
        assert stream.look() == Token(1, "block_end", "%}")
        assert stream.current.type == "name"
        assert stream.next_if("name") == Token(1, "name", "a")
        assert stream.current.type == "block_end"
        stream.push(Token(1, "name", "b"))
        assert next(stream).type == "block_end"
        assert stream.expect("name") == Token(1, "name", "b")
        assert stream.eos
        assert not stream


class TestParserLayer:
    def test_parse_from_token_stream(self, env):
        source = "{% if a %}{{ b }}{% endif %}"
        stream = Lexer(LexerConfig()).tokenize(source, None, None)
        from_stream = Parser(env, stream).parse()
        assert from_stream.dump() == env.parse(source).dump()

    def test_parse_without_environment(self, env):
        source = "{% for i in seq %}{{ i }}{% endfor %}"
        stream = Lexer(LexerConfig()).tokenize(source, None, None)
        template = Parser(None, stream).parse()
        assert template.dump() == env.parse(source).dump()

    def test_parse_source_without_environment_raises(self):
        with pytest.raises(TypeError):
            Parser(None, "{{ 1 }}")

    def test_parser_registers_no_symbols(self, env):
        """The syntax layer only produces a node tree.  Symbol
        registration happens later, in the analysis layer.
        """
        template = env.parse("{% set x = 1 %}{{ x }}")
        assert not hasattr(template, "symbols")
        assert not hasattr(template, "refs")
        assert not hasattr(template, "loads")


class TestSymbolAnalysisLayer:
    def test_if_branch_store_other_branch_load(self, env):
        """A variable stored in one if branch and loaded in another is
        resolved at the enclosing frame.  This is the scenario from the
        bytecode cache mismatch report.
        """
        template = env.parse("{% if a %}{% set x = 1 %}{% else %}{{ x }}{% endif %}")
        sym = symbols_for_node(template)
        assert sym.refs == {"a": "l_0_a", "x": "l_0_x"}
        assert sym.loads == {
            "l_0_a": ("resolve", "a"),
            "l_0_x": ("resolve", "x"),
        }
        assert sym.dump_stores() == {"x": "l_0_x"}

    def test_overlay_scope_standalone(self):
        overlay = nodes.OverlayScope(
            nodes.Const({}),
            [nodes.Assign(nodes.Name("v", "store"), nodes.Const(1))],
        )
        sym = symbols_for_node(overlay)
        assert sym.refs == {"v": "l_0_v"}
        assert sym.loads == {"l_0_v": ("undefined", None)}
        assert sym.dump_stores() == {"v": "l_0_v"}

    def test_overlay_scope_with_nested_block(self):
        overlay = nodes.OverlayScope(
            nodes.Const({}),
            [
                nodes.Assign(nodes.Name("w", "store"), nodes.Const(2)),
                nodes.Block(
                    "b", [nodes.Output([nodes.Name("w", "load")])], False, False
                ),
            ],
        )
        sym = symbols_for_node(overlay)
        assert sym.refs == {"w": "l_0_w"}
        assert sym.loads == {"l_0_w": ("undefined", None)}
        assert sym.dump_stores() == {"w": "l_0_w"}

    def test_call_block_standalone(self):
        call_block = nodes.CallBlock(
            nodes.Call(nodes.Name("m", "load"), [], [], None, None),
            [nodes.Name("z", "param")],
            [],
            [nodes.Output([nodes.Name("z", "load")])],
        )
        sym = symbols_for_node(call_block)
        assert sym.refs == {"z": "l_0_z"}
        assert sym.loads == {"l_0_z": ("param", None)}
        assert sym.dump_stores() == {"z": "l_0_z"}

    def test_block_nesting_from_parsed_tree(self, env):
        template = env.parse(
            "{% block b %}{% set y = 1 %}{% endblock %}"
            "{% call(z) m() %}{{ z }}{% endcall %}"
        )
        sym = symbols_for_node(template)
        # Block and CallBlock bodies are separate frames; only the call
        # target is visible at the template level.
        assert sym.refs == {"m": "l_0_m"}
        assert sym.loads == {"l_0_m": ("resolve", "m")}


class TestOptimizerLayer:
    unfolded = (
        "nodes.Template([nodes.Output([nodes.Add(nodes.Const(1),"
        " nodes.Mul(nodes.Const(2), nodes.Const(3)))])])"
    )
    folded = "nodes.Template([nodes.Output([nodes.Const(7)])])"

    def test_tree_before_and_after_folding(self, env):
        template = env.parse("{{ 1 + 2 * 3 }}")
        assert template.dump() == self.unfolded
        folded = optimize(template, env)
        assert folded.dump() == self.folded

    def test_optimize_without_environment(self, env):
        template = env.parse("{{ 1 + 2 * 3 }}")
        assert optimize(template).dump() == self.folded

    def test_folded_tree_compiles_and_renders(self, env):
        folded = optimize(env.parse("{{ 1 + 2 * 3 }}"), env)
        template = Template.from_code(env, env.compile(folded), env.globals)
        assert template.render() == "7"


class TestCompilePipelineRegression:
    source = (
        "{% if a %}{% set x = 'X' %}{% endif %}"
        "{% if not a %}{{ x|default('unset') }}{% endif %}"
        "|{% if a %}{{ x }}{% endif %}"
    )

    def test_bccache_hit_matches_miss(self, tmp_path):
        """Rendering must be byte-identical whether the bytecode cache
        is hit or not, also for variables defined in one if branch and
        used in another.
        """
        cache_dir = str(tmp_path)
        plain_env = Environment(loader=DictLoader({"t": self.source}))
        miss_env = Environment(
            loader=DictLoader({"t": self.source}),
            bytecode_cache=FileSystemBytecodeCache(cache_dir),
        )
        hit_env = Environment(
            loader=DictLoader({"t": self.source}),
            bytecode_cache=FileSystemBytecodeCache(cache_dir),
        )

        for a in (True, False):
            expected = plain_env.get_template("t").render(a=a)
            miss = miss_env.get_template("t").render(a=a)
            hit = hit_env.get_template("t").render(a=a)
            assert miss == expected
            assert hit == expected
