import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')

import apps.gui.test_eqn as test_code
from apps.gui.docstring_crawler import get_class_code, get_class_function_names
from apps.gui.gui_elements import MyBasicOrganizer
from library.model.model import *

# Function to update code_text based on selected checkboxes
def update_code_text(event):
    selected_functions = [checkbox.name for checkbox in checkboxes if checkbox.value]
    code_text = "\n\n".join([get_class_code(ShallowWater, func_name=func) for func in selected_functions])
    editor.value = code_text


def update_editor(cls):
    editor.value = get_class_code(cls)


def update_terminal_from_editor(button, terminal, editor):
    terminal.value = editor.value


# latex = pn.pane.LaTeX(r"""
# $\begin{aligned}
#   \nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
#   \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
#   \nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
#   \nabla \cdot \vec{\mathbf{B}} & = 0
# \end{aligned}
# $""", styles={'font-size': '18'})
def wrap_latex(text):
    begin = r'$\begin{aligned}'
    end = r'\end{aligned}$'
    output = f'{begin} {text} {end}'
    return output
#latex = pn.pane.LaTeX(wrap_latex(test_code.get_model()))
#latex = pn.pane.LaTeX(wrap_latex(model.get_latex()))
latex_text =  r"""
# WIP: Model Display

## PDE

$$
\partial_t {\mathbf{Q}} + \nabla \cdot {\mathbf{F}}({\mathbf{Q}}) + {\underline{\underline{NC}}}({\mathbf{Q}}) : \nabla {\mathbf{Q}} = {\mathbf{S}}({\mathbf{Q}})
$$

with 

$$
\mathbf{Q} = ...
$$

$$
\mathbf{F}(\mathbf{Q}) = ...
$$

$$
\underline{\underline{NC}}(\mathbf{Q}) = ...
$$

$$
\mathbf{S}(\mathbf{Q}) = ...
$$

## Boundary Conditions

$$
\begin{aligned}
& \mathbf{Q}|_{\mathbf{x}=\text{left}} = ... \\
& \mathbf{Q}|_{\mathbf{x}=\text{right}} = ...
\end{aligned}
$$

## Initial Conditions

$$
\mathbf{Q}(t=0) = ...
$$

"""
latex = pn.pane.Markdown(latex_text)


# main = GridStack(sizing_mode='stretch_both', min_height=60, min_width=190)
# main[0:9, 0:6] = editor
# main[0:9, 6:12] = latex
# main[9:12, 0:6] = pn.Spacer(styles=dict(background='purple'))
# main[9:12, 6:12] = pn.Spacer(styles=dict(background='yellow'))

#model_editor = MyBasicOrganizer(main)


# checkbox_1 = pn.widgets.Checkbox(name='Show Model')
# checkbox_2 = pn.widgets.Checkbox(name='Show Eigenvalues')
# controls = pn.Column('# Editor controls', checkbox_1, checkbox_2)
# model_editor.attach_controls(controls)


# import inspect
# import panel as pn
# from panel.layout.gridstack import GridStack
# from gui.docstring_crawler import get_class_code
# from library.model.model import ShallowWater

# Get all function names from the ShallowWater class
function_names = get_class_function_names(ShallowWater)

# Create a checkbox for each function
#values = [function_names == 'flux']
checkboxes = [pn.widgets.Checkbox(name=func_name) for func_name in function_names]

# Add event listeners to checkboxes
for checkbox in checkboxes:
    checkbox.param.watch(update_code_text, 'value')

# Create a CodeEditor widget
code_text = get_class_code(ShallowWater, func_name='flux')
editor = pn.widgets.CodeEditor(value=code_text, sizing_mode='stretch_width', language='python', theme='monokai', height=250)

# Create a layout for the checkboxes
checkbox_layout = pn.Column(*checkboxes)
button_latex = pn.widgets.Button(name='Update Latex', button_type='primary')

controls = pn.Column('# Model controls', button_latex, checkbox_layout)

main = GridStack(sizing_mode='stretch_both', min_height=60, min_width=190)
main[0:8, 0:6] = editor
main[0:8, 6:12] = latex

model_editor = MyBasicOrganizer(main)


model_editor.attach_controls(controls)


# Serve the panel
# pn.serve(main)
