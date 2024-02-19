from setuptools import setup

package_name = 'acados_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='factslabegmc',
    maintainer_email='evannsmcuadrado@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = acados_mpc.my_node:main',
            'cython_work = acados_mpc.cython_work:main',
            'ros2_integration = acados_mpc.ros_integration:main',
            'px4_integration = acados_mpc.px4_integation:main',
            'deq_mpc = acados_mpc.px4_deq_mpc:main',
            'NR_all = acados_mpc.nr_DEQ:main',
        ],
    },
)
