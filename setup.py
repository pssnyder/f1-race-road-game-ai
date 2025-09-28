from setuptools import setup, find_packages

setup(
    name="f1-race-road-game-ai",
    version="0.1.0",
    description="Educational F1 race game AI using DQN",
    packages=find_packages(include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[],  # use requirements.txt for pinned deps
)
