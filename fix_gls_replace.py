import os
import re

dir_path = '/home/sergiosvieira/Documentos/uecetex2'

replacements = [
    (r'Vehicular Edge Computing\s*\(\\gls\{VEC\}\)', r'\\gls{VEC}'),
    (r'Deep Reinforcement Learning\s*\(\\gls\{DRL\}\)', r'\\gls{DRL}'),
    (r'Internet dos Veículos\s*\(\\gls\{IoV\}\)', r'\\gls{IoV}'),
    (r'Aprendizado por Reforço\s*\(\\gls\{DRL\}\)', r'\\gls{DRL}'),
    (r'Aprendizado por Reforço Profundo\s*\(\\gls\{DRL\}\)', r'\\gls{DRL}'),
    (r'Computação de Borda Móvel\s*\(\\gls\{MEC\}\)', r'\\gls{MEC}'),
    (r'Processo de Decisão de Markov\s*\(\\gls\{MDP\}\)', r'\\gls{MDP}'),
    (r'Computação de Borda Veicular\s*\(\\gls\{VEC\}\)', r'\\gls{VEC}'),
    (r'Unidades de Beira de Estrada\s*\(\\gls\{RSU\}\)', r'\\gls{RSU}'),
    (r'Modelos de Linguagem de Grande Escala\s*\(\\glspl\{LLM\}\)', r'\\glspl{LLM}'),
    (r'Tomada de Decisão Multicritério\s*\(\\gls\{MCDM\}\)', r'\\gls{MCDM}'),
    (r'Aprendizado de Máquina\s*\(\\gls\{ML\}\)', r'\\gls{ML}'),
    (r'Aprendizado Federado\s*\(\\gls\{FL\}\)', r'\\gls{FL}'),
    (r'Abordagens baseadas em Grafos\s*\(\\gls\{GNN\}\)', r'Abordagens baseadas em \\gls{GNN}'),
    (r'União Internacional de Telecomunicações\s*\(\\gls\{ITU\}\)', r'\\gls{ITU}'),
    (r'Banda Larga Móvel Aprimorada\s*\(\\gls\{eMBB\}\)', r'\\gls{eMBB}'),
    (r'Comunicações Massivas do Tipo Máquina\s*\(\\gls\{mMTC\}\)', r'\\gls{mMTC}'),
    (r'Internet das Coisas\s*\(\\gls\{IoT\}\)', r'\\gls{IoT}'),
    (r'Comunicações Ultra-Confiáveis e de Baixa Latência\s*\(\\gls\{URLLC\}\)', r'\\gls{URLLC}'),
    (r'Nova Rede de Acesso de Rádio\s*\(\\gls\{NG-RAN\}\)', r'\\gls{NG-RAN}'),
    (r'Núcleo de Próxima Geração\s*\(\\gls\{5GC\}\)', r'\\gls{5GC}'),
    (r'Arquitetura Baseada em Serviços\s*\(\\gls\{SBA\}\)', r'\\gls{SBA}'),
    (r'Plano de Controle e do Plano de Usuário\s*\(\\gls\{CUPS\}\)', r'\\gls{CUPS}'),
    (r'Redes Definidas por \\textit\{Software\}\s*\(\\gls\{SDN\}\)', r'\\gls{SDN}'),
    (r'Virtualização de Funções de Rede\s*\(\\gls\{NFV\}\)', r'\\gls{NFV}'),
    (r'Computação de Borda de Múltiplo Acesso\s*\(\\gls\{MEC\}\)', r'\\gls{MEC}'),
    (r'Unidade Central\s*\(\\gls\{CU\}\)', r'\\gls{CU}'),
    (r'Unidades Distribuídas\s*\(\\glspl\{DU\}\)', r'\\glspl{DU}'),
    (r'modelo Não-Autônomo\s*\(\\gls\{NSA\}\)', r'modelo \\gls{NSA}'),
    (r'Não-Autônomo\s*\(\\gls\{NSA\}\)', r'\\gls{NSA}'),
    (r'Autônomo\s*\(\\gls\{SA\}\)', r'\\gls{SA}'),
    (r'Novo Rádio\s*\(\\gls\{NR\}\)', r'\\gls{NR}'),
    (r'Frequência Range 1\s*\(\\gls\{FR1\}\)', r'\\gls{FR1}'),
    (r'Frequência Range 2\s*\(\\gls\{FR2\}\)', r'\\gls{FR2}'),
    (r'Multiplexação por Divisão de Frequências Ortogonais\s*\(\\gls\{OFDMA\}\)', r'\\gls{OFDMA}'),
    (r'camada Física\s*\(\\gls\{PHY\}\)', r'\\gls{PHY}'),
    (r'Aprendizado por Reforço Multiagente\s*\(\\gls\{MARL\}\)', r'\\gls{MARL}'),
    (r'Processos de Decisão de Markov Parcialmente Observáveis\s*\(\\gls\{POMDP\}\)', r'\\gls{POMDP}'),
    (r'Repetição de Experiência Priorizada\s*\(\\gls\{PER\}\)', r'\\gls{PER}'),
    (r'Otimização de Política Proximal\s*\(\\gls\{PPO\}\)', r'\\gls{PPO}'),
    (r'Gradiente de Política Determinístico Profundo\s*\(\\gls\{DDPG\}\)', r'\\gls{DDPG}'),
    (r'Ator-Crítico Suave\s*\(\\gls\{SAC\}\)', r'\\gls{SAC}'),
]

for root, dirs, files in os.walk(dir_path):
    if '.git' in root or 'build' in root: continue
    for file in files:
        if file.endswith('.tex'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for pat, repl in replacements:
                new_content = re.sub(pat, repl, new_content, flags=re.IGNORECASE)
            
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated {filepath}")

