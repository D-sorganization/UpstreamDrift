import json

c1_p = r'C:\Users\diete\Repositories\Worktrees\UpstreamDrift-simscape-tour-runtime\scratch\staged_impact_best_checkpoint.json'
c2_p = r'C:\Users\diete\Repositories\Worktrees\UpstreamDrift-simscape-tour-runtime\scratch\candidate_staged_impact_1233s_package.json'

with open(c1_p, 'r') as f:
    d1 = json.load(f)
with open(c2_p, 'r') as f:
    d2 = json.load(f)

print(f"--- Best Checkpoint (Eval {d1.get('evaluation')}) ---")
print(f"Cost:                {d1.get('cost', 0):.4e}")
print(f"Best Cost:           {d1.get('best_cost', 0):.4e}")
print(f"Clubhead RMSE:       {d1.get('best_clubhead_rmse_mm', 0):.2f} mm")
print(f"Early RMSE:          {d1.get('early_rmse_mm', 0):.2f} mm")
print(f"Yaw Error:           {d1.get('yaw_err_1233_pct', 0):.2f} %")

print(f"\n--- Final Package ---")
print(f"Candidate ID:        {d2.get('candidate_id')}")
print(f"Clubhead Term RMSE:  {d2.get('clubhead_terminal_rmse_m', 0) * 1000.0:.2f} mm")
print(f"Early RMSE:          {d2.get('early_rmse_m', 0) * 1000.0:.2f} mm")
print(f"Whole Window RMSE:   {d2.get('rmse_m', 0) * 1000.0:.2f} mm")
print(f"Terminal RMSE:       {d2.get('terminal_rmse_m', 0) * 1000.0:.2f} mm")
print(f"Yaw Error:           {d2.get('pelvis_yaw_1233_error_pct', 0):.2f} %")
print(f"Shaft Inc Diff:      {d2.get('shaft_inclination_diff_deg', 0):.2f} deg")
print(f"Gates Passed:        {d2.get('gates_passed')}")
for g, p in d2.get('gates', {}).items():
    print(f"  {g:<22}: {'PASS' if p else 'FAIL'}")
