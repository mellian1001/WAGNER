:- use_module(library(sockets)).
:- use_module(library(charsio)).
:- use_module(library(lists)).
:- use_module(library(format)).
:- use_module(library(reif)).
:- use_module(library(between)).

run_python_code(Command, Result) :-
    socket_client_open('127.0.0.1':8888, Stream, [eof_action(eof_code), type(text)]),
    write(Stream, Command),
    flush_output(Stream),    
    get_n_chars(Stream, _, Result).

run_python_file(File_name, Result) :-
    socket_client_open('127.0.0.1':8888, Stream, [eof_action(eof_code), type(text)]),
    append("[FILE]", File_name, Send),
    write(Stream, Send),
    flush_output(Stream),    
    get_n_chars(Stream, _, Result).

% e.g. "[[1,2],[3,4]]" -> [[1,2],[3,4]]
python_string_to_term(Python_string, List) :-
    append(Python_string, ".", String),
    read_from_chars(String, List).

% e.g. "hello\n" -> '"hello\\n"'
string_to_python_string(Atom, Result) :-
    once(char_replace('\n', ['\\', 'n'], Atom, Py_str1)),
    once(char_replace('\t', ['\\', 't'], Py_str1, Py_str2)),
    once(char_replace('\r', ['\\', 'r'], Py_str2, Py_str3)),
    once(char_replace('\f', ['\\', 'f'], Py_str3, Py_str4)),
    once(char_replace('\v', ['\\', 'v'], Py_str4, Py_str5)),
    once(char_replace('\"', ['\\', '"'], Py_str5, Py_str)),
    append(['\"'], Py_str, Temp),    
    append(Temp, ['\"'], Quoted_str), 
    atom_chars(Result, Quoted_str).  

%  '123' -> 123
python_string_to_number(Python_string, Number) :-
    number_chars(Number, Python_string).

py_isdefined(Var) :-
    run_python_code("print(dir())", Res),
    python_string_to_term(Res, Ls),
    memberchk(Var, Ls).

py_showdefined(Ls) :-
    run_python_code("print(dir())", Res),
    python_string_to_term(Res, Ls).
    

% e.g. py_using("numpy")
py_using(Package) :-
    % write_term_to_chars(Package, [], Pack),
    append("import ", Package, Command),
    run_python_code(Command, Res),
    Res = [].

get_value(Var, X) :-
    py_isdefined(Var) -> 
    (   
        write_term_to_chars(Var, [], Var_str),
        append("print(", Var_str, Py_code1),
        append(Py_code1, ")", Py_code),
        run_python_code(Py_code, Res),
        catch(python_string_to_term(Res, X), error(syntax_error(_), _), X = Res)
    )
    ;
        throw(error(existence_error(_, Var), 'variable is not defined in python')). 

call_py_func(Func, Arg, X) :-
    % py_isdefined(Func),
    Ls = [Func | Arg],
    Term =.. Ls,
    write_term_to_chars(Term, [], Chars),
    append("print(", Chars, Chars1),
    append(Chars1, ")", Py_code),
    run_python_code(Py_code, X).
    % catch(python_string_to_term(Res, X), error(_, _), X = Res).

% e.g. set_array_values(arr, [0,0], 1) -> arr[0][0] = 1
set_array_value(Arr, Index, X) :-
    unify1(Arr, Index, Py_code1),
    append(Py_code1, "=", Py_code2),
    write_term_to_chars(X, [], Py_code3),
    append(Py_code2, Py_code3, Py_code),
    run_python_code(Py_code, Res),
    Res = [].

get_array_value(Arr, Index, X) :-
    unify1(Arr, Index, Py_code1),
    append("print(", Py_code1, Py_code2),
    append(Py_code2, ")", Py_code),
    run_python_code(Py_code, Res),
    catch(python_string_to_term(Res, X), error(syntax_error(_), _), X = Res).

get_tensor_value(Arr, Index, X) :-
    unify1(Arr, Index, Py_code1),
    append("print(", Py_code1, Py_code2),
    append(Py_code2, ".item())", Py_code),
    run_python_code(Py_code, Res),
    catch(python_string_to_term(Res, X), error(syntax_error(_), _), X = Res).



unify1(Ls, Index, X) :-
    py_isdefined(Ls),
    write_term_to_chars(Ls, [], Ls_str),
    unify2(Ls_str, Index, X).

unify2(Ls, [], Ls).
unify2(Ls, [Head | Rest], Res) :- 
    append(Ls, "[", Ls1),
    write_term_to_chars(Head, [], Head_str),
    append(Ls1, Head_str, Ls2),
    append(Ls2, "]", Ls3),
    unify2(Ls3, Rest, Res).


char_replace(_, _, [], []).

char_replace(X, Y, [X|T], Result) :-
    char_replace(X, Y, T, R),
    append(Y, R, Result).

char_replace(X, Y, [H|T], [H|Result]) :-
    X \= H,
    char_replace(X, Y, T, Result).


:- set_prolog_flag(answer_write_options, [max_depth(10000)]).
