# Vendored Third-Party Assets

The JavaScript files in this directory are vendored copies of third-party libraries
maintained by their respective upstream projects. Deep member-chain traversal within
these files reflects idiomatic patterns of those libraries and is outside the scope
of this project's Law-of-Demeter policy.

| File                                      | Upstream source                                                     | Notes                             |
| ----------------------------------------- | ------------------------------------------------------------------- | --------------------------------- |
| `jquery.js`                               | [jQuery](https://jquery.com/)                                       | Sphinx ships this; do not edit.   |
| `_sphinx_javascript_frameworks_compat.js` | [Sphinx](https://www.sphinx-doc.org/)                               | Compatibility shim; do not edit.  |
| `searchtools.js`                          | [Sphinx](https://www.sphinx-doc.org/)                               | Search index logic; do not edit.  |
| `sphinx_highlight.js`                     | [Sphinx](https://www.sphinx-doc.org/)                               | Syntax highlighting; do not edit. |
| `js/badge_only.js`                        | [sphinx-rtd-theme](https://github.com/readthedocs/sphinx_rtd_theme) | RTD badge; do not edit.           |
| `js/theme.js`                             | [sphinx-rtd-theme](https://github.com/readthedocs/sphinx_rtd_theme) | RTD theme; do not edit.           |
| `js/versions.js`                          | [sphinx-rtd-theme](https://github.com/readthedocs/sphinx_rtd_theme) | Version flyout; do not edit.      |

To upgrade these files, regenerate the Sphinx docs or update the theme package rather
than editing the files directly.
