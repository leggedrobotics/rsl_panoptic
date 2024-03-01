from setuptools import setup, find_packages

setup(
    name="mask2former",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for Mask2Former model implementations in robotics and machine learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithub/mask2former",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        "numpy",
        "torch",
        "torchvision",
        # etc.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
