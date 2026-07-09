# 1.0.0 (2026-07-09)


### Bug Fixes

* _provider_to_tts_source — voice discovery por tipo (piper vs openai) ([a678f24](https://github.com/Jota-project/jota-wake-trainer/commit/a678f249bad6ae7149b8528c00e3b995d886e2e3))
* añadir configs.local/ a .gitignore ([83329d8](https://github.com/Jota-project/jota-wake-trainer/commit/83329d8251457c73ea14e830b6400a7705c5c3ce))
* añadir stub _wizard_add_provider — implementación completa en Task 3 ([9f55c71](https://github.com/Jota-project/jota-wake-trainer/commit/9f55c7146f57d90c6af8f3de00763f893f5f4084))
* corregir extra openwakeword, añadir egg-info a .gitignore ([1ca5451](https://github.com/Jota-project/jota-wake-trainer/commit/1ca5451f3f365406d8779a842dfe403cc3c369a0))
* corregir README — modelo ok_jota aún no entrenado ([4c6ddbf](https://github.com/Jota-project/jota-wake-trainer/commit/4c6ddbf83dab5b06b6efbd6361493f3e05302aed))
* guard speeds vacío en ProviderConfig; eliminar parámetro project no usado ([df9283c](https://github.com/Jota-project/jota-wake-trainer/commit/df9283c3abdaad9c7a1bc52d227e1a06c6979fb0))
* state — proteger _from_dict de mutación y create_project de overwrite ([ec9cc19](https://github.com/Jota-project/jota-wake-trainer/commit/ec9cc198e5c04099309e136138efec33b1422128))


### Features

* añadir piper_downloader con fetch_voices_index y download_voice ([565c311](https://github.com/Jota-project/jota-wake-trainer/commit/565c311b4969c819f532b312f8952d39276c0d8e))
* CLI — entry point Typer con subcomandos y delegación al wizard ([18221bd](https://github.com/Jota-project/jota-wake-trainer/commit/18221bd38bd6f9292cf8c2d876eaa2b3d13026a6))
* CLI providers — subcomandos list, add, remove con flags y wizard ([74a8e68](https://github.com/Jota-project/jota-wake-trainer/commit/74a8e6854e2b8a4af076c45019cfddb31d681c78))
* comando providers piper-voices para descargar modelos Piper ([3342b34](https://github.com/Jota-project/jota-wake-trainer/commit/3342b34e009620d7bd03615c6fe1cbf8e32d8814))
* dataset calculator — estima muestras totales con augmentación ([98615d5](https://github.com/Jota-project/jota-wake-trainer/commit/98615d5d03b64435054bbd14de1d5f276af39870))
* estructura inicial de jota-wake-trainer ([c191fb3](https://github.com/Jota-project/jota-wake-trainer/commit/c191fb3a2fbc2516284c0de3b4f7e71bb6554692))
* evaluator — métricas de precisión, recall y falsos positivos ([f0dcdac](https://github.com/Jota-project/jota-wake-trainer/commit/f0dcdacd35928477a734b3e777c9fb73b1dadeea))
* extraer entrenamiento y evaluación a trainer/workflows/training.py ([fe473b7](https://github.com/Jota-project/jota-wake-trainer/commit/fe473b7351c807596aac7e30e2c28ba5950abcb0))
* extraer grabación e importación a trainer/workflows/recording.py ([e84bb35](https://github.com/Jota-project/jota-wake-trainer/commit/e84bb355e3a89dc147203e23a441d382dbb6b3c0))
* extraer selección de voces a trainer/ui/voice_selection.py ([be880d1](https://github.com/Jota-project/jota-wake-trainer/commit/be880d12dbfbb76db62f7450b6329326dbf863e1))
* extraer síntesis y provider wizard a trainer/workflows/synthesis.py ([9f36cb0](https://github.com/Jota-project/jota-wake-trainer/commit/9f36cb0dd6f5574571cdd5b6f4682c27f19d7633))
* extraer tabla de providers a trainer/ui/tables.py ([d6569ae](https://github.com/Jota-project/jota-wake-trainer/commit/d6569aee82c5a5c3992c084a100898eba4273dcb))
* importer — importación y validación de WAVs externos ([f1300d8](https://github.com/Jota-project/jota-wake-trainer/commit/f1300d8dd81fe14f492ffe802327df91af54300e))
* migración de estructura ok_jota plana a projects/ok_jota/ ([751e735](https://github.com/Jota-project/jota-wake-trainer/commit/751e735a2d73773efd41379cbaa5b0c654785630))
* providers — CRUD de providers TTS globales en configs/providers.local.json ([c032e0f](https://github.com/Jota-project/jota-wake-trainer/commit/c032e0f3f1e05baeb545bcaa5162e27e22a38ba1))
* re-exportar funciones públicas desde trainer/workflows/__init__.py ([b0d4f30](https://github.com/Jota-project/jota-wake-trainer/commit/b0d4f30a0d5013ffaf051a6c8f2c69e79bc7c217))
* recorder — captura de audio guiada con countdown y validación ([3b292c9](https://github.com/Jota-project/jota-wake-trainer/commit/3b292c90a9e329d2164a86c521884d452d8fc848))
* state — persistencia de proyectos en session.json ([dee336b](https://github.com/Jota-project/jota-wake-trainer/commit/dee336b1742225ec264292dabb65724a9a88680b))
* synthesizer — Piper TTS y endpoints OpenAI-compatible con selección de voces ([9a6c4f4](https://github.com/Jota-project/jota-wake-trainer/commit/9a6c4f47679173a77fdb70fb108697280272bea6))
* trainer core — wrapper de openWakeWord con recolección de clips ([c33185c](https://github.com/Jota-project/jota-wake-trainer/commit/c33185c0c78abd660190aa97d35a27ab468a6227))
* UI — componentes Rich para paneles, tablas y prompts contextuales ([bdbbfbd](https://github.com/Jota-project/jota-wake-trainer/commit/bdbbfbdecb79ae8444f7166879b481913d5e9d6b))
* wizard — flujo guiado completo con planificación, grabación, síntesis y entrenamiento ([d4fb07b](https://github.com/Jota-project/jota-wake-trainer/commit/d4fb07b9877baf03127ffa2cb3c55c13fc60947f))
* wizard — integración de providers globales en configuración de síntesis ([82a8ecf](https://github.com/Jota-project/jota-wake-trainer/commit/82a8ecf9238a8483c4125626e402e28c647faaec))

# Changelog

Todas las versiones publicadas de este proyecto se documentan aquí automáticamente vía [semantic-release](https://github.com/semantic-release/semantic-release), a partir de los títulos de las PRs mergeadas a `main` (convención de [Conventional Commits](https://www.conventionalcommits.org/)).

No edites este fichero a mano — semantic-release lo reescribe en cada release.
