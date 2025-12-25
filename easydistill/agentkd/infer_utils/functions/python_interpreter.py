import io
import multiprocessing
import types
import ast
import pickle
from contextlib import redirect_stdout
from typing import Dict, Any
from ast import unparse


class MockTools:
    @staticmethod
    def final_answer_print(answer):
        """Only for final answer"""
        print(answer)

def _get_serializable_globals(globals_dict: dict, executed_code: str, previous_scope: dict = None) -> dict:
    """
    Filter a dictionary to only include serializable objects.
    For functions and import statements, extract their source code by parsing the original code string.
    """
    serializable_globals = {}
    
    # Start with functions from previous scope
    if previous_scope:
        for key, value in previous_scope.items():
            if isinstance(value, dict) and value.get('__type__') == 'function':
                serializable_globals[key] = value
    
    # Parse current code to extract functions and imports
    function_sources = {}
    import_statements = []
    try:
        tree = ast.parse(executed_code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_sources[node.name] = unparse(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_statements.append(unparse(node))
    except (SyntaxError, AttributeError):
        pass

    # Add import statements to the serializable state
    if previous_scope and '__imports__' in previous_scope and previous_scope['__imports__'].get('__type__') == 'import_block':
        # Merge with existing imports
        existing_imports = previous_scope['__imports__']['__sources__']
        # Combine and deduplicate
        all_imports = list(dict.fromkeys(existing_imports + import_statements))
        serializable_globals['__imports__'] = {
            '__type__': 'import_block',
            '__sources__': all_imports
        }
    else:
        serializable_globals['__imports__'] = {
            '__type__': 'import_block',
            '__sources__': import_statements
        }

    # Process variables and functions in globals
    for key, value in globals_dict.items():
        # Skip built-in variables and problematic types
        if key.startswith('__') or isinstance(value, (types.ModuleType, types.CodeType)):
            continue
        
        # Handle functions by using extracted source code
        if isinstance(value, types.FunctionType):
            if key in function_sources:
                serializable_globals[key] = {
                    '__type__': 'function',
                    '__source__': function_sources[key]
                }
            continue

        # Test if object can be pickled
        try:
            pickle.dumps(value)
            serializable_globals[key] = value
        except (pickle.PicklingError, TypeError):
            pass
            
    return serializable_globals

def _execute_code_in_process(args):
    """Helper function to execute code in a separate process"""
    code, scope_dict, output_queue = args
    try:
        # Recreate the scope from the dictionary
        scope = {}
        
        # Rebuild state: imports first, then functions, then variables
        # 1. Rebuild imports
        if '__imports__' in scope_dict and scope_dict['__imports__'].get('__type__') == 'import_block':
            for import_src in scope_dict['__imports__']['__sources__']:
                try:
                    exec(import_src, scope)
                except Exception:
                    pass

        # 2. Rebuild functions from previous executions
        for key, value in scope_dict.items():
            if key == '__imports__': 
                continue  # Skip imports as they're already processed

            if isinstance(value, dict) and value.get('__type__') == 'function':
                try:
                    # Execute function definition in scope
                    exec(value['__source__'], scope)
                except Exception:
                    pass

        # 3. Load regular variables
        for key, value in scope_dict.items():
            if key == '__imports__': 
                continue
                
            # Skip functions as they're already processed
            if isinstance(value, dict) and value.get('__type__') == 'function':
                continue
                
            # Copy other serializable variables
            try:
                pickle.dumps(value)
                scope[key] = value
            except (pickle.PicklingError, TypeError):
                pass

        # Add mock tools to the scope
        scope['final_answer_print'] = MockTools.final_answer_print
        
        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            exec(code, scope)
        output = output_stream.getvalue()
        if not output:
            output = "Execution successful, no output."
            
        # Filter globals to only include serializable objects
        serializable_scope = _get_serializable_globals(scope, code, scope_dict)
                
        output_queue.put({"output": output, "updated_scope": serializable_scope, "error": None})
    except Exception as e:
        output_queue.put({"output": f"Error: {e}", "updated_scope": scope_dict, "error": str(e)})

def python_interpreter(code: str, scope: Dict[str, Any], timeout: int = 10) -> Dict:
    """Execute Python code with a timeout mechanism using multiprocessing."""
    # Prepare scope for serialization
    prepared_scope = {}
    
    # Copy over the existing scope, handling special objects properly
    for key, value in scope.items():
        prepared_scope[key] = value
    
    # Create a queue for communication
    output_queue = multiprocessing.Queue()
    
    # Package arguments for the process
    args = (code, prepared_scope, output_queue)
    
    # Create and start the process
    process = multiprocessing.Process(target=_execute_code_in_process, args=(args,))
    process.start()
    
    try:
        # Wait for the result with timeout
        result = output_queue.get(timeout=timeout)
        process.join()  # Wait for process to finish
    except multiprocessing.TimeoutError:
        # Terminate the process if it times out
        process.terminate()
        process.join()
        result = {"output": f"Error: Code execution timed out after {timeout} seconds", 
                  "updated_scope": scope, 
                  "error": "timeout"}
    except Exception as e:
        # Handle other exceptions
        process.terminate()
        process.join()
        result = {"output": f"Error: {e}", 
                  "updated_scope": scope, 
                  "error": str(e)}
    
    return {"output": result["output"], "updated_scope": result["updated_scope"]}
