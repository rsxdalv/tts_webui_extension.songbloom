import setuptools
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setuptools.setup(
    name="tts_webui_extension.songbloom",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="Your Name",
    description="A template extension for TTS Generation WebUI",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rsxdalv/tts_webui_extension.songbloom",
    project_urls={},
    scripts=[],
    install_requires=[
        # Add your dependencies here
        # "numpy",
        # "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
