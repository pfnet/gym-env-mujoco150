
from setuptools import setup

install_requires = [
    'gym>=0.9.2',
    'mujoco-py<1.50.2,>=1.50.1',
]

setup(
    name='gym-env-mujoco150',
    version='0.1.0',
    packages=['gym_env_mujoco150'],
    package_data={'gym_env_mujoco150': ['assets/*.xml']},
    url='https://github.com/pfnet/gym-env-mujoco150',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=install_requires,
)
