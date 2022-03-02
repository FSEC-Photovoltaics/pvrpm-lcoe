# Contributing to PVRPM

---

Thank you for taking time to contribute to PVRPM! Please refer to the following guidelines below before you submit an issue or pull request. For making edits to the PVRPM code, please follow how to install the package in edit mode [here](https://pvrpm.readthedocs.io/en/latest/tutorial_1installation.html).

### Quick Reference
[Documentation](https://pvrpm.readthedocs.io/en/latest/)

[License](../LICENSE)

[Issues](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues/)

---

## Asking Questions
Before asking a question, please make sure to read the [documentation](https://pvrpm.readthedocs.io/en/latest/) carefully, as your answer to basic questions will be there. If you can not find the answer to your question, check all of issues, both open and closed, [here](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues?q=) to see if someone else asked a similar question that was resolved. If you still can not find your question, please create a new [issue](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues/) and use the **question issue template** for submitting a question. Questions should be clear and concise, and contain as much information as possible; the more information provided the better an answer can be.  

## Feature Requests
Feature requests be submitted as issues [here](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues) and should follow the **feature request template**.
Feature requests should be concise, specific, and if possible include examples of how the feature should work. It will then be assigned a priority level by one of the members of the repository.

## Reporting Bugs
Bug reports should also be submitted as issues [here](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues) and follow the **bug report template**. Bug reports need to provide the operating system and hardware specifications of the computer that ran into the bug, full stack trace (use the `--trace` flag when running a simulation to get stack traces), and include a `zip` file containing **your PVRPM `YAML` configuration, `JSON` files for the SAM case, and the weather file used with the simulation.** This will allow others to reproduce and confirm the bug on other systems.

## Contributing Code
Contribution to the code base of PVRPM should be done with pull requests from a forked repository of PVRPM (see [GitHub's pull request guide](https://docs.github.com/en/pull-requests)). Contributions **must solve a current bug or feature request in PVRPM!** Pull requests that do not have an associated bug or enhancement in an open issue [here](https://github.com/FSEC-Photovoltaics/pvrpm-lcoe/issues) will not be merged or considered until one is opened and referenced to the pull request.

#### Code style
PVRPM follows [black](https://github.com/psf/black) code formatter for all Python code. **You must format you code properly using black before submitting a PR!** Your code can be properly formatted by running these commands in your Python environment:

```bash
$ pip install black
$ cd /path/to/pvrpm/repo
$ python -m black -l 120 .
```

#### Testing
Before submitting, you can easily test your new additions by running the integrated tests in PVRPM using `pytest`. To run the tests, please follow the installation for testing [here](https://pvrpm.readthedocs.io/en/latest/tutorial_1installation.html) then run:

```bash
$ pytest
```

Depending on your hardware, tests should take around 20 minutes to complete. If your tests include new features in PVRPM that are not configured in the `pytest` configuration file, you may edit the test simulation file at `tests/integration/case/test.yml` to add new the features you implemented. **All tests must pass before submitting a PR!**

Again, thank you for taking the time to make PVRPM better! **Make sure to follow templates when contributing! Issues not following templates may receive delayed responses or be automatically closed.**
