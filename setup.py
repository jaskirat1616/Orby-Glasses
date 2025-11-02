#!/usr/bin/env python3
"""
Setup script for OrbyGlasses PyPI distribution.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='orbglasses',
    version='0.9.0',
    description='AI-powered navigation assistant for blind and visually impaired users',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='OrbyGlasses Contributors',
    author_email='',  # Add email if desired
    url='https://github.com/jaskirat1616/Orby-Glasses',
    project_urls={
        'Bug Tracker': 'https://github.com/jaskirat1616/Orby-Glasses/issues',
        'Documentation': 'https://github.com/jaskirat1616/Orby-Glasses',
        'Source Code': 'https://github.com/jaskirat1616/Orby-Glasses',
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['config/*.yaml'],
    },
    install_requires=read_requirements(),
    python_requires='>=3.10,<3.13',
    entry_points={
        'console_scripts': [
            'orbglasses=src.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='accessibility navigation computer-vision blind assistive-technology SLAM',
    license='GPL-3.0-or-later',
)
