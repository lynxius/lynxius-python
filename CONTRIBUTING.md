# Contribution Guidelines

## Code Style We Follow

The Python code style used by Lynxius projects generally follows PEP8.

In addition to the standards outlined in PEP 8, we have a few guidelines:

#### Max line length

We set the max line length to 88 characters.

#### Import order and grouping

Imports at the beginning of the code should be grouped into three groups in this order:

1. Python's builtin modules
2. Third-party modules
3. The modules from this project

Leave an empty line to separate each import group.

#### docstring format

 A typical docstring should follow this [Google style](http://google.github.io/styleguide/pyguide.html#381-docstrings):

    """[Summary]

    Args:
        path (str): The path of the file to wrap
        field_storage (FileStorage): The :class:`FileStorage` instance to wrap
        temporary (bool): Whether or not to delete the file when the File
        instance is destructed

    Returns:
        BufferedFileStorage: A buffered writable file descriptor
    """

## Lint And Format Your Code

We reccomand setting up `flake8` and `black` to style your Python code.

### Setup Flake8 And Black on VSCode

* Make sure to install [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) and [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) VsCode extensions.

* Add the snippet below to your `.vscode/settings.json`:

  ```json
  {
    // General
    "editor.rulers": [
        88
    ],
    "editor.tabSize": 2,
    "editor.insertSpaces": true,
    "editor.formatOnSave": true,
    // Flake8
    "flake8.args": [
        "--max-line-length=88"
    ],
    // Black
    "black-formatter.args": [
        "--line-length=88"
    ],
    // HTML
    "[html]": {
        "editor.tabSize": 2
    },
    // Javascript
    "[javascript]": {
        "editor.tabSize": 2
    },
    // CSS
    "[css]": {
        "editor.tabSize": 2
    },
    // TypeScript
    "[typescript]": {
        "editor.tabSize": 2
    },
    // React
    "[typescriptreact]": {
        "editor.tabSize": 2
    },
    // Python
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "never",
            "source.organizeImports.ruff": "explicit",
        }
    }
  }
  ```