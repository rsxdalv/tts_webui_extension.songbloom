import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.songbloom",
    packages=setuptools.find_namespace_packages(),
    version="0.0.2",
    author="rsxdalv",
    description="A template extension for TTS Generation WebUI",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.songbloom",
    project_urls={},
    scripts=[],
    install_requires=[
        "tts-webui.songbloom @ git+https://github.com/rsxdalv/tts-webui.songbloom@main",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
