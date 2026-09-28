import os

from setuptools import find_packages, setup


def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.isfile(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return 'Causal Action Tokenizer'


setup(
    name='catok',
    version='0.1.0',
    author='Chenyu Zhang',
    author_email='chenyuzhangx@gmail.com',
    description='Causal Action Tokenizer: a diffusion-based action tokenizer for robot learning',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ChenyuZhangX/CausalActionTokenizer',
    packages=find_packages(exclude=['tests*', 'docs*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
