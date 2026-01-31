:- use_module('./scpl_client.pl').

run(Env_name) :-
    py_using("prolog_call_tools"),
    init_env(Env_name),
    get_stone_pickaxe(Env_name).

get_stone_pickaxe(Env) :- 
    get_stone(Env),
    craft_stone_pickaxe(Env).

craft_stone_pickaxe(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.call_stone_pickaxe', [Env_py], Res),
    Res = "true".

get_stone(Env) :- 
    get_wood_pickaxe(Env),
    mine_stone(Env).

mine_stone(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.call_stone', [Env_py], Res),
    Res = "true".

get_wood_pickaxe(Env) :- 
    collect_wood(Env),
    craft_wood_pickaxe(Env).

craft_wood_pickaxe(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.call_wood_pickaxe', [Env_py], Res),
    Res = "true".

collect_wood(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.call_wood', [Env_py], Res),
    Res = "true".

init_env(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.init_env', [Env_py], Res),
    Res = "true".

to_py_string([], []).
to_py_string(Ls_string, Ls_py_string) :-
    Ls_string = [Head_string | Ls_rest_string],
    Ls_py_string = [Head_py_string | Ls_rest_py_string],
    string_to_python_string(Head_string, Head_py_string),
    to_py_string(Ls_rest_string, Ls_rest_py_string).