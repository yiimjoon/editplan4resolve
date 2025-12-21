from VideoForge.broll.matcher import QueryGenerator


def test_query_generator_outputs():
    gen = QueryGenerator(top_n=6)
    result = gen.generate("Shooting 4K footage of NewYork Skyline!!!")
    assert "primary" in result
    assert "fallback" in result
    assert "boost" in result
    assert result["fallback"]
    assert result["primary"]
