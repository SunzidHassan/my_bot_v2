from setuptools import find_packages, setup

package_name = 'my_bot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='sunzidhassan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "my_bot_footballer_1_llm= my_bot_controller.my_bot_footballer_v1_llm:main",
            "my_bot_subpub= my_bot_controller.my_bot_subpub:main",
            "my_bot_footballer_v2_llmrl=my_bot_controller.my_bot_footballer_v2_llmrl_1:main"
        ],
    },
)
