from setuptools import find_packages, setup
from typing import List

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_list = []
    with open("requirements.txt", "r") as f:
        requirement_list = f.readlines()
        requirement_list = [requirement.replace("\n", "") for requirement in requirement_list]
        if "-e ." in requirement_list:
            requirement_list.remove("-e .")
    return requirement_list
setup(
    name="qubicml"
    ,version="0.0.1"
    ,description="Qubit State Discrimination"
    ,long_description=long_description
    ,long_description_content_type="text/markdown"
    ,author="Neel Vora"
    ,author_email='NRVora@lbl.gov'
    ,packages=find_packages()
    ,install_requires=requirements()
    ,python_requires='>=3.9'
    ,include_package_data=True
    ,package_data={
		"": ["*.json"],
		}

)