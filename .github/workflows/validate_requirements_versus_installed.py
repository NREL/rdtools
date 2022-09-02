
import subprocess
import pkg_resources
import json


def parse_requirements_file(fn):
    with open(fn, 'r') as f:
        requirements = pkg_resources.parse_requirements(f.read())
    packages = [
        {'name': req.name, 'version': str(req.specifier)[2:]}
        for req in requirements
    ]
    return packages


def assert_equal(installed, required):
    # for now only check package names, not versions
    installed = {req['name'] for req in installed}
    required = {req['name'] for req in required}
    missing = installed - required
    extra = required - installed
    if missing or extra:
        msg = (
            'Requirements files do not match installed versions:\n'
            f'Missing dependencies: {missing}\n'
            f'Extra dependencies: {extra}'
        )
        raise AssertionError(msg)


out = subprocess.check_output(['pip', 'list', '--format', 'json'])
installed_packages = json.loads(out.decode())

requirements1 = parse_requirements_file(r'requirements.txt')
requirements2 = parse_requirements_file(r'docs/notebook_requirements.txt')
requirements = requirements1 + requirements2

assert_equal(installed_packages, requirements)
