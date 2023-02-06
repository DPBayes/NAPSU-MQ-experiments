repeats = 20
epsilons = [0.1, 0.3, 0.5, 1.0]
# repeats = 2
# epsilons = [0.5, 1.0]
algorithms = [
    "max-ent", "PGM", "PGM-repeat-5", "PGM-repeat-10", "PGM-repeat-20", "RAP",
    "PEP"
]

rule all:
    input:
        results="results/us-census/results.csv",
        runtimes="results/us-census/runtime.csv",
        rhats="results/us-census/rhats.csv",
        fail_counts="results/us-census/fail-counts.csv",
        marginal_accuracies="results/us-census/marginal-accuracy-results.csv",

rule prepare_data:
    input:
        "datasets/USCensus1990.data.txt"
    output:
        "datasets/us-census/us-census-reduced.csv"
    threads: 1
    resources:
        mem_mb=1000,
        time="00:10:00"
    script:
        "../scripts/us_census/prepare-data.py"

rule orig_result:
    input:
        "datasets/us-census/us-census-reduced.csv"
    output:
        "datasets/us-census/orig_result.p"
    threads: 1
    resources:
        mem_mb=1000,
        time="00:10:00"
    script:
        "../scripts/us_census/orig-result.py"

rule synth_data:
    input:
        data="datasets/us-census/us-census-reduced.csv",
    output:
        "results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p"
    threads: 4
    resources:
        mem_mb=1000,
        time=lambda wildcards: "1-00:00:00" if wildcards.algo == "max-ent" else "1:00:00",
        partition="short"#lambda wildcards: "medium" if wildcards.algo == "max-ent" else "short"
    script:
        "../scripts/us_census/synth-data.py"

rule extract_rhats:
    input:
        "results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p"
    output:
        "results/us-census/diagnostics/{algo}_{repeat_ind}_{epsilon}_rhat.csv"
    threads: 1
    resources:
        mem_mb=4000,
        time="00:10:00"
    script:
        "../scripts/adult-reduced-extract-rhats.py"

rule extract_fail_counts:
    input:
        "results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p"
    output:
        "results/us-census/diagnostics/{algo}_{repeat_ind}_{epsilon}_fail_count.csv"
    threads: 1
    resources:
        mem_mb=4000,
        time="00:10:00"
    script:
        "../scripts/extract-laplace-restarts.py"

rule merge_fail_counts:
    input: 
        expand(
            "results/us-census/diagnostics/{algo}_{repeat_ind}_{epsilon}_fail_count.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=["max-ent"]
        )
    output:
        "results/us-census/fail-counts.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=10,
        time="00:10:00",
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"
    
rule merge_rhats:
    input: 
        expand(
            "results/us-census/diagnostics/{algo}_{repeat_ind}_{epsilon}_rhat.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=["max-ent"]
        )
    output:
        "results/us-census/rhats.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=10,
        time="00:10:00",
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"

rule analysis:
    input:
        syn_data="results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p",
        orig_result="datasets/us-census/orig_result.p"
    threads: 1
    resources:
        mem_mb=4000,
        time="00:10:00",
    output:
        "results/us-census/analysis/{algo}_{repeat_ind}_{epsilon}.csv"
    script:
        "../scripts/us_census/analysis.py"

rule merge_analyses:
    input: 
        expand(
            "results/us-census/analysis/{algo}_{repeat_ind}_{epsilon}.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=algorithms
        )
    output:
        "results/us-census/results.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=10,
        time="00:10:00",
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"

rule marginal_accuracy:
    input:
        syn_data="results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p",
        data="datasets/us-census/us-census-reduced.csv",
    threads: 1
    resources: 
        mem_mb=5000,
        time="01:00:00"
    output:
        "results/us-census/marginal-accuracy/{algo}_{repeat_ind}_{epsilon}.csv"
    script:
        "../scripts/us_census/marginal-accuracy.py"

rule merge_marginal_accuracies:
    input: 
        expand(
            "results/us-census/marginal-accuracy/{algo}_{repeat_ind}_{epsilon}.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=algorithms
        )
    output:
        "results/us-census/marginal-accuracy-results.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"

rule runtime_report:
    input:
        expand(
            "results/us-census/synth_data/{algo}_{repeat_ind}_{epsilon}.p",
            epsilon=epsilons, repeat_ind=range(repeats), algo=algorithms
        )
    output:
        csv="results/us-census/runtime.csv",
        latex="latex/us-census-runtime.tex"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    script:
        "../scripts/us_census/runtime-report.py"

rule report:
    input:
        results="results/us-census/results.csv",
        rhats="results/us-census/rhats.csv",
        marginal_accuracy="results/us-census/marginal-accuracy-results.csv",
        orig_result="datasets/us-census/orig_result.p"
    log: 
        notebook="processed_report-us-census.py.ipynb"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    notebook:
        "../scripts/us_census/report.py.ipynb"

