@echo off
rem Three reference clips for the video -> ARDY test (Wan 2.1 T2V 1.3B, ~5 s each).
cd /d %~dp0
set PY=venv_t2v\Scripts\python.exe
%PY% t2v_wan.py --name pirate --seed 1 --prompt "An exhausted pirate captain in a long dark coat and a tricorn hat stands on a wooden ship deck, leans forward with both hands as if on a railing, then raises one hand to shade his eyes and slowly scans the horizon from left to right, shoulders slumped with fatigue."
%PY% t2v_wan.py --name thief --seed 2 --prompt "A nervous thief in dark hooded clothes stands idle in an empty stone courtyard, glances quickly over his left shoulder and then his right, shifts his weight from foot to foot, crouches slightly and checks the corners around him."
%PY% t2v_wan.py --name noble --seed 3 --prompt "An arrogant nobleman in an elegant embroidered coat stands in a palace hall, lifts his chin, waves one hand dismissively in front of him, puts the other hand on his hip and turns his body away with disdain."
