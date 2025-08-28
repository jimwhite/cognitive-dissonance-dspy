#!/usr/bin/env python3
"""
CI/CD incremental verification for pull requests.

Only verifies functions that changed in the PR, saving CI time.
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

from formal_verification.incremental_verifier import IncrementalVerifier


def get_changed_files(base_branch: str, pr_branch: str) -> list:
    """Get list of changed Python files in PR."""
    cmd = f"git diff --name-only {base_branch}...{pr_branch} -- '*.py'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split('\n') if f]


def main():
    parser = argparse.ArgumentParser(description='Incremental verification for CI')
    parser.add_argument('--base-branch', required=True, help='Base branch (e.g., main)')
    parser.add_argument('--pr-branch', required=True, help='PR branch')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='Minimum proof coverage threshold')
    args = parser.parse_args()
    
    print(f"🔍 Incremental Verification: {args.pr_branch} → {args.base_branch}")
    print("=" * 60)
    
    # Get changed files
    changed_files = get_changed_files(args.base_branch, args.pr_branch)
    if not changed_files:
        print("✅ No Python files changed - skipping verification")
        return 0
    
    print(f"📝 Changed files: {len(changed_files)}")
    for f in changed_files[:5]:  # Show first 5
        print(f"   - {f}")
    if len(changed_files) > 5:
        print(f"   ... and {len(changed_files) - 5} more")
    print()
    
    # Run incremental verification
    verifier = IncrementalVerifier()
    
    total_verified = 0
    total_proven = 0
    failed_files = []
    
    for file_path in changed_files:
        if not Path(file_path).exists():
            continue
        
        print(f"🔧 Verifying {file_path}...")
        
        with open(file_path, 'r') as f:
            code = f.read()
        
        results = verifier.verify_incremental(code, force_full=False)
        
        verified = results['functions_verified']
        proven = results['functions_proven']
        
        total_verified += verified
        total_proven += proven
        
        if verified > 0:
            coverage = proven / verified
            status = "✅" if coverage >= args.threshold else "❌"
            print(f"   {status} Coverage: {coverage:.1%} ({proven}/{verified} functions)")
            
            if coverage < args.threshold:
                failed_files.append((file_path, coverage))
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    
    if total_verified > 0:
        overall_coverage = total_proven / total_verified
        print(f"Total functions verified: {total_verified}")
        print(f"Total functions proven: {total_proven}")
        print(f"Overall coverage: {overall_coverage:.1%}")
        
        if overall_coverage >= args.threshold:
            print(f"\n✅ Verification PASSED (coverage >= {args.threshold:.0%})")
        else:
            print(f"\n❌ Verification FAILED (coverage < {args.threshold:.0%})")
            print("\nFailed files:")
            for file_path, coverage in failed_files:
                print(f"   - {file_path}: {coverage:.1%}")
            return 1
    else:
        print("No functions found to verify")
    
    # Save state for next run
    status = verifier.get_verification_status()
    print(f"\n📁 Verification state saved (version {status['version']})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())