# repeats = 100
repeats = 2
# epsilons = [0.1, 0.25, 0.5, 1, 10, 100]
epsilons = [0.5, 1]
algorithms = [
    "max-ent-all-queries", 
    "max-ent-no-noise-aware", 
    "rap-all-queries", 
    "PGM",
    "PEP",
    "PrivBayes",
    "privLCM"
]

localrules: all 
rule all:
    input: 
        toy_data_results="results/toy-data/results.csv",
        toy_data_fail_counts="results/toy-data/fail-counts.csv",

ruleorder: synth_data_privLCM > synth_data
    
rule synth_data:
    output:
        "results/toy-data/synth_datasets/synth_data_{algo}_{repeat_ind}_{epsilon}.p"
    threads: 2
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    script:
        "../scripts/generate_synth_data.py"

rule synth_data_privLCM_generation:
    output:
        ["results/toy-data/privLCM_temp/privLCM_{{repeat_ind}}_{{epsilon}}_{syn_dataset_ind}.csv".format(syn_dataset_ind=i) for i in range(100)]
    threads: 1
    shell:
        "Rscript workflow/scripts/run_privLCM.r {wildcards.epsilon} {wildcards.repeat_ind} results/toy-data/privLCM_temp/"

rule synth_data_privLCM:
    input:
        ["results/toy-data/privLCM_temp/privLCM_{{repeat_ind}}_{{epsilon}}_{syn_dataset_ind}.csv".format(syn_dataset_ind=i) for i in range(100)]
    output:
        "results/toy-data/synth_datasets/synth_data_privLCM_{repeat_ind}_{epsilon}.p"
    threads: 1
    script:
        "../scripts/privLCM_syn_data_to_pickle.py"
        
rule analysis:
    input:
        "results/toy-data/synth_datasets/synth_data_{algo}_{repeat_ind}_{epsilon}.p"
    output:
        "results/toy-data/analysis/analysis_{algo}_{repeat_ind}_{epsilon}.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    script:
        "../scripts/run_analysis.py"
        
rule merge_analyses:
    input: 
        expand(
            "results/toy-data/analysis/analysis_{algo}_{repeat_ind}_{epsilon}.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=algorithms
        )
    output:
        "results/toy-data/results.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=10,
        time="00:10:00"
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"

rule extract_fail_counts:
    input:
        "results/toy-data/synth_datasets/synth_data_{algo}_{repeat_ind}_{epsilon}.p"
    output:
        "results/toy-data/diagnostics/{algo}_{repeat_ind}_{epsilon}_fail_count.csv"
    threads: 1
    resources:
        mem_mb=1000,
        time="00:10:00"
    script:
        "../scripts/extract-laplace-restarts.py"

rule merge_fail_counts:
    input: 
        expand(
            "results/toy-data/diagnostics/{algo}_{repeat_ind}_{epsilon}_fail_count.csv",
            epsilon=epsilons, repeat_ind=range(repeats), algo=["max-ent-all-queries", "max-ent-no-noise-aware"]
        )
    output:
        "results/toy-data/fail-counts.csv"
    threads: 1
    resources:
        nodes=1,
        mem_mb=10,
        time="00:10:00",
    shell: 
        "{{ head -n 1 {input[0]}; tail -q -n +2 {input}; }} > {output}"
    
rule report:
    input:
        results="results/toy-data/results.csv",
        fail_counts="results/toy-data/fail-counts.csv"
    log: 
        notebook="processed_report-toy-data.py.ipynb"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:10:00"
    notebook:
        "../scripts/report.py.ipynb"

rule compressed_results:
    input: "results/toy-data/results.csv"
    output: "toy-data/results.zip"
    threads: 1
    resources:
        nodes=1,
        mem_mb=100,
        time="00:30:00"
    shell: "zip -r -q results.zip results"
