def pytest_configure(config):
    config.addinivalue_line("markers", "integration: usa llama-server e modelos reais (skip se ausentes)")
