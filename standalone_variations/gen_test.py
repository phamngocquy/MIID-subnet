import asyncio, json      
from types import SimpleNamespace as NS
from MIID.validator.query_generator import QueryGenerator 
cfg = NS()
neuron = NS()                                                                                           
# Set to True to skip complex LLM generation and judge; uses default template
neuron.use_default_query = False
# If you want to use your local Ollama for judge/generation, set these:
neuron.ollama_url = "http://localhost:11434"
neuron.ollama_request_timeout = 15
cfg.neuron = neuron

# Seed config sizes
cfg.seed_names = NS(sample_size=15)

async def main():
    qg = QueryGenerator(cfg)
    seeds, template, labels, gen_model, gen_timeout, judge_model, judge_timeout, gen_log = await qg.build_queries()
    print("--- TEMPLATE ---")
    print(template)
    print("\n--- LABELS ---")
    print(json.dumps(labels, ensure_ascii=False, indent=2))
    print("\n--- FIRST 3 SEEDS ---")
    print(json.dumps(seeds[:3], ensure_ascii=False, indent=2))
    print("\n--- GENERATION LOG KEYS ---")
    print(list(gen_log.keys()))

asyncio.run(main())