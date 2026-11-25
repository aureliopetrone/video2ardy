@echo off
rem Two more reference clips (Wan 2.1 T2V 1.3B).
cd /d %~dp0
set PY=venv_t2v\Scripts\python.exe
%PY% t2v_wan.py --name drunk --seed 4 --prompt "A drunk sailor in a striped shirt staggers on a wooden pier, sways from side to side, almost falls forward, catches his balance with his arms spread wide and laughs."
%PY% t2v_wan.py --name captain --seed 5 --prompt "A proud ship captain in a blue coat stands on the deck giving orders, points forward energetically with his right arm, then stamps his foot and waves his crew on with both arms."
