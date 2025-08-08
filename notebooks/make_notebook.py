import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor

def make_student_notebook(notebook_path, output_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Iterate through cells
    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if '# YCH' in line:
                    indent = ' ' * (len(line) - len(line.lstrip(' ')))
                    # Find the part after # YCH: 
                    placeholder = line.split('# YCH: ')[-1].strip()

                    # Find the variable it is assigned to, if any
                    if '=' in line:
                        var_name = line.split('=')[0].strip()
                        line = f"{var_name} = YourCodeHere(\"{placeholder}\")  # Your code here."
                    else:
                        line = "YourCodeHere(\"" + placeholder + "\")  # Your code here."
                    new_lines.append(indent + line)
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)

    preprocessor = ClearOutputPreprocessor()
    nb, _ = preprocessor.preprocess(nb, {})

    # Save modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":

    # make_student_notebook('tutorial_gp_master.ipynb', 'student_versions/tutorial_gp_student.ipynb')
    make_student_notebook('tutorial_bo_master.ipynb', 'student_versions/tutorial_bo_student.ipynb')
