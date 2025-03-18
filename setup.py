from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
	long_description = f.read()

setup(
	name='PPT',
	version='0.0.1',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	description='PPT Debugging version',
	author='Jaeho Kim, Kwangryeol Park',
	author_email='kjh3690@unist.ac.kr, pkr7098@unist.ac.kr',
	url='https://github.com/jaeho3690/PPT_PIP',
	license='MIT',
	python_requires='>=3.4',
	install_requires=['torch', 'einops', 'numpy<2.0', 'tqdm'],
	packages=find_packages(),
 	keywords=['PPT', 'Permutation', 'Pytorch', 'Transformer', 'Time-series', 'Self-supervised'],
)