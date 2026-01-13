def test_jinja_no_autoescape():
    from ab_openapi_python_generator.language_converters.python.jinja_config import (
        create_jinja_env,
    )

    env = create_jinja_env()
    template = env.from_string("{{ s }}")
    out = template.render(s="'scope' : scope,")
    assert "&#39;" not in out
    assert out == "'scope' : scope,"
