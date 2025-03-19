from setuptools import find_packages, setup

def get_requirement()->list[str]:

    """
    This function will return list of requirements
    """
    requirement_list:list[str] = []


    
    return requirement_list

setup(
    name = "sensor",
    version= "0.0.1",
    author="Nitesh",
    author_email="nit51196@gmail.com",
    packages= find_packages(),
    install_requires=get_requirement(),
)