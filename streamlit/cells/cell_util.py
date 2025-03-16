import importlib
import sys

sys.path.append("cells")

def setup_globals(conf):
    """Runs through all the configs and executes any entries that should be processed as determined by having the key 'ss_globals_fn'. This makes sure
    that cells that the user does not open will have created/modified the 
    relevant global variable in ss. 

    parameters:
        conf: dict with configuration, e.g. default.json in configs directory
    """
    print("+++Running cells.cell_util::setup_globals")
    for page in conf.keys():
        if page == "tabs":
            continue
        for cell_i, cell in enumerate(conf[page]):
            print(f'     setup_globals importing: {cell["import"]}')
            module = importlib.import_module(cell["import"])
            fn_call = f"module.setup_globals('{page}', '{cell_i}')"
            try:
                print(f"      setup_globals running: {fn_call}")
                eval(fn_call)
                print(f"      setup_globals run")
            except AttributeError as e:
                print(f"      no setup_globals defined")
                continue

def create_and_display_stats(Context, df):    
    questions = df['question_num'].unique()
    raw_N_match = 0
    ans_N_match = 0
    repeat_count = len(df['run'].unique())
    correct_on_run = [0] * repeat_count
    for question in questions:
        raw = df[df['question_num'] == question]['raw_response'].unique()
        if len(raw) == 1:
            raw_N_match += 1
        ans = df[df['question_num'] == question]['pred'].unique()
        if len(ans) == 1:
            ans_N_match += 1
        else:
            print(f"answer variation {question}")
            Context.markdown(f"Answer variation {question}")
        for i, row in df[df['question_num'] == question].iterrows():
            if row['pred'] == row['gt']:
                correct_on_run[row['run']] += 1

    accuracy_on_run = [corr/len(questions) for corr in correct_on_run]
    Context.markdown(accuracy_on_run)
    Context.markdown((f"TARr@{repeat_count}: {raw_N_match/len(questions):.1%} ="
                + f"{raw_N_match}/{len(questions)}"))
    Context.markdown((f"TARa@{repeat_count}: {ans_N_match/len(questions):.1%} ="
                + f"{ans_N_match}/{len(questions)}"))


def render(conf, page):
    """Renders entries in .json configuration file, converted to a dict, by 
    eval on string of indicated call. 
    """
    print("***Running cells.cell_util::render")
    if page not in conf:
        raise ValueError(f"Page '{page}' not found in config")
    page_conf = conf[page]
    print(f"  render config entry: {page} with {page_conf}")
    for cell_i, cell in enumerate(page_conf):
        print(f"    importing cell: {cell['import']}")
        module = importlib.import_module(cell["import"])
        args = []
        for arg in cell.get("args", []):
            if isinstance(arg, dict):
                args.append(str(arg["var"]))
            else:
                args.append("'" + str(arg) + "'")
        if "fn" in cell:
            function_call = "module." + cell["fn"] + "(" + ", ".join(args) + ")"
            print(f"      render running: {function_call}")
            eval(function_call)

def parse_args(args: list) -> dict:
    """Parses command line arguments and has defaults as indicated.
    Python argparse does not work with pytest so this is a simple replacement
    Arguments:
        args: list: invocation arguments used when streamlit invoked
                    'streamlit run Foo.py db=local, config=dev.json'
    Returns:
        dict: specified args with defaults as shown
    Raises:
        RunTimeError: If there is an apparent '=' argument that is not expected.
    """
    arg_vals = {"db": "ec2", "config": None}
    for arg in args:
        if "=" in arg:
            (k, v) = arg.split("=")
            if k in ["db", "config"]:
                arg_vals[k] = v
            else:
                raise RuntimeError(f"Unrecognized argument {k} in command line")
    return arg_vals