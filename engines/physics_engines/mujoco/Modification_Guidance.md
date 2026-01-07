# Jules Architecture Upgrade Guide v2.1

**Objective:** Upgrade the "Control Tower" architecture to use the official **Jules Tools CLI** (`@google/jules`) and **REST API**. This improves stability, enables parallel task execution, and aligns with Google's long-term support for the agent.

## 1\. Prerequisites

  * **API Key:** Ensure you have a valid Jules API Key.
  * **Secrets:** Update your GitHub Repository Secrets:
      * Replace `JULES_API_TOKEN` with `JULES_API_KEY` (The new CLI uses `x-goog-api-key` semantics).
  * **Environment:** Ensure `node-version: '20'` is used in all workflows (already present in your template).

-----

## 2\. Workflow Upgrades (The Hands)

Replace the legacy `jules task fix` commands in your worker workflows with the official `jules remote new` syntax.

### A. Auto-Repair (`.github/workflows/Jules-Auto-Repair.yml`)

**Change:** Switch to `remote new` and let Jules handle the branch management.

```yaml
      - name: Trigger Jules Auto-Repair
        if: steps.verify.outputs.branch != ''
        env:
          JULES_API_KEY: ${{ secrets.JULES_API_KEY }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          TARGET_RUN_ID: ${{ inputs.run_id }}
        run: |
          BRANCH="${{ steps.verify.outputs.branch }}"
          gh run view $TARGET_RUN_ID --log-failed > logs.txt
          
          # New CLI Syntax
          # We use --repo to target this specific repository context
          jules remote new \
            --repo ${{ github.repository }} \
            --session "CI Failure on branch '$BRANCH'. Fix these errors: $(cat logs.txt)" \
            --branch "$BRANCH"
```

### B. Test Generator (`.github/workflows/Jules-Test-Generator.yml`)

**Change:** Use the **`--parallel`** flag to generate multiple test options simultaneously. This is a massive upgrade for quality assurance.

```yaml
      - name: Generate Tests (Parallel Mode)
        if: steps.files.outputs.files != ''
        env:
          JULES_API_KEY: ${{ secrets.JULES_API_KEY }}
          FILES: ${{ steps.files.outputs.files }}
        run: |
          # Request 3 parallel sessions. Jules will try 3 different testing strategies.
          jules remote new \
            --repo ${{ github.repository }} \
            --session "Generate comprehensive pytest unit tests for: [ $FILES ]. Focus on edge cases." \
            --branch ${{ github.ref_name }} \
            --parallel 3
```

### C. Review Responder (`.github/workflows/Jules-Review-Fix.yml`)

**Change:** Simplify the auth and execution steps.

```yaml
      - name: Execute Repairs
        if: steps.feedback.outputs.run_repair == 'true'
        env:
          JULES_API_KEY: ${{ secrets.JULES_API_KEY }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: |
          jules remote new \
            --repo ${{ github.repository }} \
            --session "Address code review feedback on PR #$PR_NUMBER: $(cat feedback.txt)" \
            --pr $PR_NUMBER
```

-----

## 3\. Orchestrator Optimization (The Brain)

**File:** `.github/workflows/Jules-Control-Tower.yml`

**Optimization:** For simple triage tasks or "pings," you can skip the heavy `npm install` and use the REST API directly. This reduces the billing/runtime of the Control Tower itself.

**Example Implementation (Optional Replacement for Triage Job):**

```yaml
  triage:
    runs-on: ubuntu-latest
    if: github.actor != 'jules-bot'
    steps:
      - name: Fast API Ping (Optional)
        env:
          API_KEY: ${{ secrets.JULES_API_KEY }}
        run: |
          # Example: Just checking if Jules is alive or valid before dispatching expensive workers
          curl -s -H "x-goog-api-key: $API_KEY" \
            https://jules.googleapis.com/v1alpha/sessions?pageSize=1
```

-----

## 4\. Render Integration (Deployment Healing)

If you deploy this project to Render (e.g., if you add a web dashboard for the metrics), add this specialized worker.

**New File:** `.github/workflows/Jules-Render-Healer.yml`

```yaml
name: Jules Render Healer
on:
  deployment_status:
    state: [failure]

jobs:
  fix-deploy:
    if: github.event.deployment.environment == 'production'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install -g @google/jules
      
      - name: Heal Deployment
        env:
          JULES_API_KEY: ${{ secrets.JULES_API_KEY }}
          DEPLOY_URL: ${{ github.event.deployment.target_url }}
        run: |
          jules remote new \
            --repo ${{ github.repository }} \
            --session "Production deployment failed at $DEPLOY_URL. Analyze the configuration and fix the crash."
```

-----

## 5\. Verification Checklist

  * [ ] **Secret Updated:** `JULES_API_KEY` is set in Repo Settings.
  * [ ] **CLI Version:** `npm install -g @google/jules` is fetching the latest version (no version pinning in `package.json`).
  * [ ] **Parallelism:** Check the Jules Dashboard (or `jules remote list`) to see multiple sessions created for Test Generation.
  * [ ] **PR Creation:** Verify that `jules remote new` is successfully opening PRs back to your repository.