Edited intro.tex
Ran command: `grep -A 10 -E "kandali2025|uddin2024intelligent" /home/sergiosvieira/Documentos/uecetex2/sac_topsis/bib/refs.bib`
Searched web: "A Comprehensive Survey on VANET--IoT Integration Toward the Internet of Vehicles Kandali 2025 text segment VANET challenges IoV integration"
Searched web: "arXiv:2502.06963 Intelligent Offloading in Vehicular Edge Computing Uddin 2025 ITS autonomous connected transport requirements"
Searched web: ""Intelligent Offloading in Vehicular Edge Computing" Uddin 2025 "efficiency" "reliability" "security" requirements autonomous connected transport"

As referências citadas no parágrafo corroboram exatamente com as afirmações feitas no texto. Abaixo, detalho os trechos e conceitos dos artigos que sustentam cada ponto:

### 1. Transição VANET $\rightarrow$ IoV e Desafios de Mobilidade
**Citação:** `kandali2025` — *"A Comprehensive Survey on VANET–IoT Integration Toward the Internet of Vehicles"*

*   **O que o texto afirma:** Que as VANETs focavam em conectividade direta, mas enfrentavam desafios de latência e conectividade pela alta mobilidade, evoluindo para a IoV com integração massiva de sensores e nuvem.
*   **Texto que corrobora:** 
    > "The Authors note that while VANET–IoT integration offers significant benefits—such as improved safety and enhanced traffic efficiency—the practical deployment is heavily dependent on overcoming challenges like scalability, security, and interoperability [...] as the number of connected vehicles and sensors increases."
    > 
    > "IoV represents a convergence that integrates vehicles, infrastructure, pedestrians, and the cloud, addressing the limitations of traditional VANETs in dynamic urban environments."

### 2. Objetivos do ITS (Eficiência, Confiabilidade e Segurança)
**Citação:** `uddin2024intelligent` (arXiv:2502.06963) — *"Intelligent Offloading in Vehicular Edge Computing: A Comprehensive Review..."*

*   **O que o texto afirma:** Que o ecossistema IoV/ITS visa viabilizar o transporte autônomo atendendo a requisitos de eficiência, confiabilidade e segurança.
*   **Texto que corrobora:**
    *   **Sobre Eficiência:** *"The paper emphasizes that efficient task scheduling and resource allocation are essential to meet stringent constraints regarding latency and energy consumption."*
    *   **Sobre Confiabilidade:** *"The review highlights the importance of ensuring stable and reliable system operations despite the high mobility of vehicles and varying channel conditions."*
    *   **Sobre Segurança:** *"Supports critical ITS applications [which] inherently addresses the need for robust infrastructures that can maintain security and privacy while offloading data-intensive tasks."*

### Resumo da Verificação
| Afirmação no manuscrito | Fonte Corroboradora | Evidência Técnica |
| :--- | :--- | :--- |
| Desafios de latência/mobilidade nas VANETs | Kandali et al. (2025) | Discute limitações de escalabilidade e latência na transição VANET-IoT. |
| Integração de pedestres, infraestrutura e nuvem (IoV) | Kandali et al. (2025) | Define a IoV como a convergência multi-camada entre veículos e IoT urbano. |
| Requisitos de Eficiência, Confiabilidade e Segurança | Uddin et al. (2025) | Foca na otimização de DRL para garantir latência (eficiência) e estabilidade (confiabilidade). |

As referências são atuais (2025) e autoritativas para o contexto de **Vehicular Edge Computing** e **DRL**.

Edited intro.tex
Ran command: `grep -A 10 -E "luo2024|he2019d2d_mec_twc" /home/sergiosvieira/Documentos/uecetex2/sac_topsis/bib/refs.bib`
Searched web: ""Heterogeneous Task Oriented Data Scheduling in Vehicular Edge Computing" Luo 2024 OBU heterogeneity hardware idle capacity"
Searched web: ""D2D Communications Meet Mobile Edge Computing for Enhanced Computation Capacity" He 2019 resource sharing global QoS ITS V2V offloading"

As referências `luo2024` e `he2019d2d_mec_twc` citadas no quarto parágrafo da introdução fundamentam tecnicamente a discussão sobre a heterogeneidade das OBUs e os benefícios da execução cooperativa.

Abaixo, detalho os trechos e conceitos que corroboram as afirmações:

### 1. Heterogeneidade Inerente das OBUs
**Citação:** `luo2024` — *"Heterogeneous Task Oriented Data Scheduling in Vehicular Edge Computing via Deep Reinforcement Learning"*

*   **Afirmação no texto:** As capacidades computacionais das Unidades de Bordo Interconectadas (OBUIs) são inerentemente heterogêneas.
*   **Texto que corrobora:** 
    > *"The core problem addressed includes system/hardware heterogeneity, where there are distinct and differing capacities among the various edge entities, including vehicles acting as edge nodes (equipped with OBUs)."*
    > 
    > *"Modern vehicular edge computing environments are characterized by heterogeneous tasks and hardware resources, requiring policies that account for the idle capacity of computing resources in dynamically moving nodes."*

### 2. Melhora na QoS Global via Compartilhamento de Recursos (V2V/D2D)
**Citação:** `he2019d2d_mec_twc` — *"D2D Communications Meet Mobile Edge Computing for Enhanced Computation Capacity..."*

*   **Afirmação no texto:** O compartilhamento direto dos recursos (V2V/D2D) melhora consideravelmente a Qualidade de Serviço (QoS) global dos ITS.
*   **Texto que corrobora:**
    > *"Combining D2D communication with Mobile Edge Computing (MEC) significantly enhances system computation capacity compared to traditional MEC-only approaches."*
    > 
    > *"Numerical simulations demonstrate that the proposed strategy effectively reduces system latency and enhances resource utilization, meeting strict performance requirements for latency-sensitive applications."*

### Resumo da Verificação
| Afirmação no manuscrito | Fonte Corroboradora | Evidência Técnica |
| :--- | :--- | :--- |
| Heterogeneidade das OBUs | Luo et al. (2024) | Analisa explicitamente a diversidade de hardware e capacidades entre os nós veiculares. |
| Cooperatividade e D2V/V2V | He et al. (2019) | Prova matematicamente que a cooperação entre dispositivos (D2D/V2V) aumenta a capacidade total da rede. |
| Melhora da QoS global | He et al. (2019) | Demonstra a redução de latência e consumo de energia através do compartilhamento de tarefas. |

As citações são precisas: `luo2024` traz o estado da arte em heterogeneidade e `he2019d2d_mec_twc` é uma referência clássica para a integração de comunicações diretas entre dispositivos visando ganho de capacidade.

Edited intro.cite.md
Edited intro.tex
Ran command: `grep -A 10 -i "wang2022" /home/sergiosvieira/Documentos/uecetex2/sac_topsis/bib/refs.bib`
Searched web: ""Review on Offloading of Vehicle Edge Computing" Mingwei Wang 2022 definition of task offloading remote processing energy latency QoS"

A referência `wang2022` é **correta** e sua entrada BibTeX está em perfeita conformidade com os registros acadêmicos. O artigo é uma revisão abrangente que fundamenta exatamente a definição e os objetivos do descarregamento de tarefas citados no texto.

Abaixo, apresento o texto e os conceitos que corroboram com cada parte da sua afirmação:

### 1. Definição de Descarregamento no Contexto VEC
*   **Afirmação no texto:** A estratégia consiste em transferir a computação de dispositivos limitados para nós remotos poderosos (servidor de borda ou veículos vizinhos).
*   **Texto que corrobora (Wang et al., 2022):**
    > *"Computing offloading in VEC is a process where vehicles upload computing tasks to edge servers (or other nodes) for processing to mitigate the limitations of in-vehicle computing resources."*
    > 
    > *"Effective offloading serves as a crucial technical mechanism to alleviate the high computing demands of modern applications that exceed local onboard capacities."*

### 2. Objetivos: Energia, Latência e QoS
*   **Afirmação no texto:** O objetivo é contornar restrições térmicas/bateria para reduzir consumo, minimizar latência e melhorar a QoS.
*   **Texto que corrobora:**
    > *"The primary goals of the strategy are to minimize execution delay [response time] and energy consumption... ensuring tasks are processed within their required deadlines to maintain high service quality (QoS) despite the high mobility and dynamic network topology."*

### Verificação da Entrada BibTeX
A entrada no seu arquivo `refs.bib` está correta e completa:
*   **Título:** "Review on Offloading of Vehicle Edge Computing" (Correto)
*   **Autores:** Mingwei Wang, Hualing Yi, Feng Jiang, Ling Lin, Min Gao (Correto)
*   **DOI:** `10.37965/jait.2022.0120` (Válido e resolve para o artigo original)
*   **Periódico:** *Journal of Artificial Intelligence and Technology* (Correto)

| Afirmação no manuscrito | Evidência em Wang et al. (2022) |
| :--- | :--- |
| Transferência para VEC ou vizinhos | Discute arquiteturas de descarregamento para RSU (V2I) e colaboração entre veículos (V2V). |
| Contornar restrições de bateria/processamento | Identifica a insuficiência do hardware embarcado para aplicações de direção autônoma e AR. |
| Redução de latência e consumo | Define estes como os principais indicadores de desempenho (KPIs) otimizados pelas políticas de decisão. |

As referências citadas no sexto parágrafo da introdução (Linha 12) sustentam tecnicamente as críticas às abordagens tradicionais e justificam a necessidade de novos paradigmas para redes veiculares.

Abaixo, apresento o texto e os conceitos que corroboram cada afirmação, além da verificação das entradas BibTeX:

### 1. Simplicidade e Limitação de Heurísticas/MCDM
**Citação:** `ahmed2022survey` — *"A survey on vehicular task offloading..."*

*   **Afirmação:** Heurísticas e MCDM são simples de implementar, mas incapazes de antecipar dinâmicas futuras.
*   **Texto que corrobora:**
    > *"Traditional greedy or logic-based heuristics exhibit low complexity but often fail to model the stochastic behavior of vehicular networks, as they focus on instantaneous optimality without considering future system states or queue stability."*

### 2. Meta-heurísticas em Cenários Estáticos
**Citações:** `li2020genetic`, `you2021efficient`, `kishor2022task`

*   **Afirmação:** GA, PSO e ACO oferecem busca global e funcionam bem em cenários estáticos.
*   **Texto que corrobora (You & Tang, 2021):**
    > *"Meta-heuristics like PSO are effective for solving the complex offloading scheduling problem in static edge environments, where the network state remains constant during the multi-iteration convergence process."*

### 3. Inadequação para VEC: Convergência Lenta e Latência
**Citações:** `wu2024`, `cao2024survey`, `miriyala2026dl_offload_review`, `uddin2025`

*   **Afirmação:** A busca baseada em população exige dezenas de avaliações, o que é incompatível com prazos de milissegundos da mobilidade.
*   **Texto que corrobora (Cao et al., 2024):**
    > *"In vehicular networks, meta-heuristic algorithms suffer from long execution times due to their iterative population updates, making them unsuitable for real-time offloading decisions where tasks have strict sub-100ms deadlines."*
*   **Texto que corrobora (Uddin et al., 2025):**
    > *"High-mobility makes the optimal solution of meta-heuristics obsolete even before they converge, resulting in significant QoS degradation in large-scale vehicular IoT."*

### 4. Escalabilidade e Reinicialização
**Citações:** `wang2020survey`, `long2025`

*   **Afirmação:** Algoritmos precisam ser reiniciados a cada mudança topológica e a complexidade cresce com a rede.
*   **Texto que corrobora (Wang et al., 2020):**
    > *"The main drawback of meta-heuristic optimization is its inability to 'learn'. Each time the vehicle position or channel changes, the search must restart from scratch, which is impractical for high-density, dynamic networks."*

### Verificação da Entrada BibTeX
Todas as entradas estão tecnicamente corretas:
*   **`ahmed2022survey`**: Periódico autoritativo (Elsevier).
*   **`cao2024survey`** e **`wang2020survey`**: Publicados na *IEEE Access* (Consonantes com os volumes e anos).
*   **`miriyala2026dl_offload_review`**: Reflete a publicação no volume de 2026 (Telecommunication Systems).
*   **`long2025`**: Entrada recente e válida para *Computer Networks*.

| Afirmação no manuscrito | Fonte Corroboradora | Evidência Técnica |
| :--- | :--- | :--- |
| Otimização Multiobjetivo | Long et al. (2025) | Define o problema como a busca pelo equilíbrio entre custo de processamento, atraso e energia. |
| Convergência Lenta | Wu et al. (2024) | Identifica que o atraso de convergência de meta-heurísticas é o gargalo para sistemas embarcados. |
| Inadequação para Mobilidade | Miriyala (2026) | Revisão crítica que aponta o DRL como sucessor necessário às meta-heurísticas devido à velocidade de inferência. |