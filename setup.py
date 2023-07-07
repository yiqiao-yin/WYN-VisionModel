from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='specimenid',
    version='0.0.0',
    author='Yiqiao Yin',
    author_email='eagle0504@gmail.com',
    description='AI Library for Vision Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yiqiao-yin/WYN-VisionModel.git',
    packages=['src/vision_models'],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tensorflow',
        'opencv-python',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)