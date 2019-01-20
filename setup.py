from setuptools import setup

setup(name='rlp',
      version='0.1',
      description='Reinforcement learning algorithm implements',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: MacOS X',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python :: 3.5',
      ],
      url='https://github.com/JZ95/RLPractical',
      author='JZhou',
      author_email='j.zhou0518@gmail.com',
      license='GPLv3',
      packages=['rlp'],
      install_requires=[
          'numpy',
          'tqdm',
          'jupyter'
      ],
      zip_safe=False)
