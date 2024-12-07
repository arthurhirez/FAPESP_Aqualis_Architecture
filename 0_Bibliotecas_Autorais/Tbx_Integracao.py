import time

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

'''
****//********//********//********//********//********//********//********//********//********//********//****
FUNÇÕES MERGE COM TOLERANCIA
****//********//********//********//********//********//********//********//********//********//********//****
'''


# Função base
def merge_with_tolerance(df1, df2, id_dfs, agg_cols, var_col, tolerance):
    # First, do an exact merge on the 'CEP' column
    df_merged = pd.merge(df1, df2, on = agg_cols, how = 'inner', suffixes = (id_dfs[0], id_dfs[1]))

    # Filter rows where the 'NUMERO_RUA' column from df1 is within the tolerance range of df2
    df_merged = df_merged[(df_merged[f'{var_col}' + id_dfs[0]] >= df_merged[f'{var_col}' + id_dfs[1]] - tolerance) &
                          (df_merged[f'{var_col}' + id_dfs[0]] <= df_merged[f'{var_col}' + id_dfs[1]] + tolerance)]

    # Reordering columns
    agg_col_suffixes = [f'{col}{id_dfs[0]}' for col in agg_cols] + [f'{col}{id_dfs[1]}' for col in agg_cols if
                                                                    f'{col}{id_dfs[1]}' in df_merged.columns]
    var_col_suffixes = [f'{var_col}{id_dfs[0]}', f'{var_col}{id_dfs[1]}']

    # Get the rest of the columns
    other_cols = [col for col in df_merged.columns if col not in agg_col_suffixes + var_col_suffixes]
    other_cols = [c for c in other_cols if c not in agg_cols]
    # New order: agg_cols + var_col_suffixes + other columns
    new_order = agg_cols + var_col_suffixes + other_cols

    # Reorder columns
    df_merged = df_merged[new_order]

    return df_merged



def merge_df_tolerance(data_cep_iptu, data_iptu, data_imoveis, tol_NUMERO = 10, tol_PVTOS = 5, tol_OBSOLENCIA = 0.1):
    qwe = merge_with_tolerance(df1 = data_cep_iptu,
                               df2 = data_iptu[['DESC_RUA', 'NUMERO_RUA', 'CEP', 'QUANTIDADE DE PAVIMENTOS',
                                                'FATOR DE OBSOLESCENCIA', 'TIPO DE PADRAO DA CONSTRUCAO',
                                                'TESTADA PARA CALCULO', 'VALOR DO M2 DO TERRENO']],
                               id_dfs = ['_cep', '_iptu'],
                               agg_cols = ['CEP'],
                               var_col = 'NUMERO_RUA',
                               tolerance = tol_NUMERO)

    qwe['QUANTIDADE DE PAVIMENTOS'] = qwe['QUANTIDADE DE PAVIMENTOS'].astype(int)

    qwe['Imovel_residencial'] = np.where(qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.contains('Residencial'), 1, 0)

    # Create 'Imovel_vertical' column: 1 if 'vertical' is in the string, 0 otherwise
    qwe['Imovel_vertical'] = np.where(qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.contains('vertical'), 1, 0)

    # Create a new column for the remaining part ('padrão X')
    qwe['Padrao'] = qwe['TIPO DE PADRAO DA CONSTRUCAO'].str.extract(r'(padrão \w)')
    qwe['Padrao'] = qwe['Padrao'].str.replace('padrão ', '')

    asd = merge_with_tolerance(df1 = qwe.rename(columns = {'DESC_RUA_cep': 'DESC_RUA',
                                                           'NUMERO_RUA_cep': 'NUMERO_RUA',
                                                           'QUANTIDADE DE PAVIMENTOS': 'Andares_tipo'}),
                               df2 = data_imoveis[
                                   ['DESC_RUA', 'NUMERO_RUA', 'Imovel_residencial', 'Imovel_vertical', 'Andares_tipo',
                                    'Total_Unidades', 'Idade_predio', 'Blocos',
                                    'M2_util_unidade_tipo', 'M2_total_unidade_tipo', 'RS_por_M2_area_util_IGPM',
                                    'RS_por_M2_area_total_IGPM', ]],
                               id_dfs = ['_cep_iptu', '_cg'],
                               agg_cols = ['DESC_RUA', 'NUMERO_RUA', 'Imovel_residencial', 'Imovel_vertical'],
                               var_col = 'Andares_tipo',
                               tolerance = tol_PVTOS).drop_duplicates()

    asd = asd[asd['Imovel_vertical'] == 1]  # muitas incertezas com horizontais

    # DataFrame containing obsolescence factors
    # df_factors = pd.DataFrame({
    #     'Idade do Prédio (em anos)': ["Menor que 1"] + list(range(1, 43)),
    #     'Fatores de Obsolescência para padrões A e B': [1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.90, 0.89,
    #                                                     0.88, 0.86, 0.84, 0.83, 0.81, 0.79, 0.78, 0.76, 0.74, 0.72,
    #                                                     0.70, 0.68, 0.66, 0.64, 0.62, 0.59, 0.57, 0.55, 0.52, 0.50,
    #                                                     0.48, 0.45, 0.42, 0.40, 0.37, 0.34, 0.32, 0.29, 0.26, 0.23,
    #                                                     0.20, 0.20, 0.20],
    #     'Fatores de Obsolescência para demais padrões e tipos': [1.00, 0.99, 0.99, 0.98, 0.97, 0.96, 0.96, 0.95, 0.94,
    #                                                              0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.88, 0.86, 0.85,
    #                                                              0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.75,
    #                                                              0.74, 0.73, 0.71, 0.70, 0.69, 0.67, 0.66, 0.64, 0.63,
    #                                                              0.62, 0.60, 0.59, 0.57, 0.56, 0.54, 0.52]
    #     })
    #
    # def get_obsolescence_factor(age, pattern, df_factors):
    #     # Map age to lookup value
    #     if age < 1:
    #         age_lookup = "Menor que 1"
    #     else:
    #         age_lookup = age
    #
    #     # Determine the column to use based on the pattern
    #     if pattern in ['A', 'B']:
    #         factor_column = 'Fatores de Obsolescência para padrões A e B'
    #     else:
    #         factor_column = 'Fatores de Obsolescência para demais padrões e tipos'
    #
    #     # Perform the lookup and handle missing results
    #     factors = df_factors[df_factors['Idade do Prédio (em anos)'] == age_lookup]
    #
    #     if not factors.empty:
    #         factor = factors[factor_column].values[0]
    #     else:
    #         # Handle the case where no matching age is found (you can adjust this as needed)
    #         factor = -1  # Or set a default factor value
    #
    #     return factor

    # Apply the function to your DataFrame
    asd['Fator_obsolescencia_calculado'] = asd.apply(
        lambda row: get_obsolescence(row['Idade_predio'], row['Padrao']), axis = 1)

    asd = asd[abs(asd['Fator_obsolescencia_calculado'] - asd['FATOR DE OBSOLESCENCIA'].astype(float)) < tol_OBSOLENCIA]

    asd['TOL_NUM'] = abs(asd['NUMERO_RUA'] - asd['NUMERO_RUA_iptu'])

    asd['TESTADA PARA CALCULO'] = asd['TESTADA PARA CALCULO'].astype(float)

    return asd[['DESC_RUA', 'NUMERO_RUA', 'NUMERO_RUA_iptu', 'Andares_tipo_cep_iptu', 'Andares_tipo_cg', 'TOL_NUM',
                'TESTADA PARA CALCULO',
                'Fator_obsolescencia_calculado', 'FATOR DE OBSOLESCENCIA', 'Total_Unidades', 'Blocos',
                'M2_util_unidade_tipo', 'M2_total_unidade_tipo', 'RS_por_M2_area_util_IGPM',
                'RS_por_M2_area_total_IGPM', 'VALOR DO M2 DO TERRENO', 'TIPO DE PADRAO DA CONSTRUCAO']].sort_values(by = ['DESC_RUA', 'NUMERO_RUA'])




# Função que faz merge e retorna casos que tiveram correspondencia devido à tolerância
def get_cases_tolerance(df1, df2, confirmed_cases, agg_cols = ['CEP'],
                        tol_NUMERO = 10, column_order = None):
    # Merge com CEP e numero rua
    ## Tolerância de +- 5 no número
    merge_TOL = merge_with_tolerance(df1 = df1, df2 = df2,
                                     id_dfs = ['', '_CEP'],
                                     agg_cols = agg_cols,
                                     var_col = 'NUMERO_RUA',
                                     tolerance = tol_NUMERO)

    print('Merge GEOEMBRAESP - SABESP')
    print(
        f"Empreendimentos de lados opostos da rua -> eliminados: {merge_TOL[(merge_TOL['NUMERO_RUA'] - merge_TOL['NUMERO_RUA_CEP']) % 2 != 0].shape[0]}")

    merge_TOL = merge_TOL[(merge_TOL['NUMERO_RUA'] - merge_TOL['NUMERO_RUA_CEP']) % 2 == 0]

    # Foi encontrada correspondencia! (mesmos casos de merge_GEO_LOGRADOURO)
    # match_cases = merge_GEO_LOGRADOURO_TOL[
    #     merge_GEO_LOGRADOURO_TOL['NUMERO_RUA'] == merge_GEO_LOGRADOURO_TOL['NUMERO_RUA_CEP']]

    # Analise mais criteriosa -> casos que não estão entre os mais confiáveis
    check_NUMERO = merge_TOL[~merge_TOL['PDE'].isin(confirmed_cases)]

    # Agregar dados para comparar com IPTU
    ## Andares tipo, idade do predio
    agg_cols = ['Empreendimento_Nome', 'CEP', 'DESC_RUA', 'Imovel_residencial', 'Imovel_vertical']

    agg_functions = {
        'Idade_predio': 'mean',
        'Blocos': 'sum',
        'Unidades_andar': 'mean',
        'Andares_tipo': 'mean',
        'Total_Unidades': 'sum',
        'Elevadores': 'sum',
        'Coberturas': 'sum',
        'Dormitorios': 'mean',
        'Banheiros_Sociais': 'mean',
        'Vagas_Garagem': 'mean',
        'M2_util_unidade_tipo': 'mean',
        'M2_total_unidade_tipo': 'mean',
        'M2_Terreno': 'mean',
        'M2_util_cobertura': 'mean',
        'M2_total_cobertura': 'mean',
        'RS_total_unidade_tipo_IGPM': 'mean',
        'RS_por_M2_area_util_IGPM': 'mean',
        'RS_por_M2_area_total_IGPM': 'mean',
        'RS_total_cobertura_IGPM': 'mean',
        }

    # Considerando numero da rua IPTU
    col_size = ['NUMERO_RUA', 'CEP']
    agg_val = ['NUMERO_RUA']
    df_TOL = check_NUMERO[agg_cols + list(agg_functions.keys()) + agg_val].drop_duplicates().copy()
    df_TOL = df_TOL.groupby(agg_cols + agg_val).agg(agg_functions).sort_values(
        ['DESC_RUA'] + agg_val).reset_index()
    print(f'{"Endereços possíveis - IPTU:":<40}{df_TOL[col_size].drop_duplicates().shape[0]}')

    # Considerando numero da rua SABESP
    agg_val = ['NUMERO_RUA_CEP']
    df_TOL_CEP = check_NUMERO[agg_cols + list(agg_functions.keys()) + agg_val].drop_duplicates().copy()

    df_TOL_CEP = df_TOL_CEP.groupby(agg_cols + agg_val).agg(agg_functions).sort_values(
        ['DESC_RUA', 'NUMERO_RUA_CEP']).reset_index()
    df_TOL_CEP = df_TOL_CEP.rename(columns = {'NUMERO_RUA_CEP': 'NUMERO_RUA'})
    print(f'{"Endereços possíveis - SABESP:":<40}{df_TOL_CEP[col_size].drop_duplicates().shape[0]}')

    # Dados finais
    df = pd.concat([df_TOL, df_TOL_CEP], axis = 0).drop_duplicates()

    df_enderecos = df[['DESC_RUA', 'NUMERO_RUA', 'CEP']].drop_duplicates()
    print(f'{"Endereços possíveis (novos):":<40}{df_enderecos[col_size].drop_duplicates().shape[0]}')

    if column_order is None: column_order = df.columns
    return df[column_order], df_enderecos


# Função que faz o pipeline completo de merge com tolerancia
def merge_df_tolerance(data_cep_iptu, data_iptu, data_imoveis, tol_NUMERO = 10, tol_PVTOS = 5, tol_OBSOLENCIA = 0.1,
                       column_order = None):
    # Merge endereços target com dados do IPTU -> TOLERANCIA NOS NUMEROS
    DATA_IPTU = merge_with_tolerance(df1 = data_cep_iptu,
                                     df2 = data_iptu,
                                     id_dfs = ['_cep', '_iptu'],
                                     agg_cols = ['CEP'],
                                     var_col = 'NUMERO_RUA',
                                     tolerance = tol_NUMERO)

    # Conversão padrao tipo de imovel
    DATA_IPTU = convert_padrao_iptu(DATA_IPTU.copy(), drop_original_col = False,
                                    compute_obsolence = False).drop_duplicates()
    DATA_IPTU['QUANTIDADE DE PAVIMENTOS'] = DATA_IPTU['QUANTIDADE DE PAVIMENTOS'].astype(float)

    # Merge resultado com dados GEOEmbraesp -> TOLERANCIA NO NUMERO DE ANDARES
    ## Ja foi computada diferenças nos números
    DATA_IPTU_GEO = merge_with_tolerance(df1 = DATA_IPTU.rename(columns = {'DESC_RUA_cep': 'DESC_RUA',
                                                                           'NUMERO_RUA_cep': 'NUMERO_RUA',
                                                                           'QUANTIDADE DE PAVIMENTOS': 'Andares_tipo'}),
                                         df2 = data_imoveis,
                                         id_dfs = ['_cep_iptu', '_cg'],
                                         agg_cols = ['DESC_RUA', 'NUMERO_RUA', 'Imovel_residencial', 'Imovel_vertical'],
                                         var_col = 'Andares_tipo',
                                         tolerance = tol_PVTOS).drop_duplicates()

    # Retornando variáveis para nome original
    DATA_IPTU_GEO.rename(columns = {'Andares_tipo_cep_iptu': 'QUANTIDADE DE PAVIMENTOS',
                                    'Andares_tipo_cg': 'Andares_tipo', }, inplace = True)
    DATA_IPTU_GEO['CEP'] = DATA_IPTU_GEO['CEP_cg']
    DATA_IPTU_GEO.drop(columns = ['CEP_cep_iptu', 'CEP_cg'], inplace = True)

    # Filtro somente para imoveis verticais (muitas incertezas com horizontais)
    DATA_IPTU_GEO = DATA_IPTU_GEO[DATA_IPTU_GEO['Imovel_vertical'] == 1]

    # Tolerancia OBSOLENCIA
    DATA_IPTU_GEO = convert_padrao_iptu(DATA_IPTU_GEO.copy(), drop_original_col = True,
                                        compute_obsolence = True).drop_duplicates()
    DATA_IPTU_GEO = DATA_IPTU_GEO[abs(
        DATA_IPTU_GEO['Fator_obsolescencia_calculado'] - DATA_IPTU_GEO['FATOR DE OBSOLESCENCIA'].astype(
            float)) < tol_OBSOLENCIA]

    # Computar diferencas nos numeros
    DATA_IPTU_GEO['TOL_NUM'] = abs(DATA_IPTU_GEO['NUMERO_RUA'] - DATA_IPTU_GEO['NUMERO_RUA_iptu'])

    if column_order is None: column_order = DATA_IPTU_GEO.columns
    return DATA_IPTU_GEO[column_order].sort_values(by = ['DESC_RUA', 'NUMERO_RUA'])


# Função que faz verificação dos resultados pipeline + scrapping
def verify_targets(df_tgts, tol_TESTADA = 0.65, scrap_tgts = None, filename_scrap = 'scraps/tol.csv',
                   return_accepted = True, column_order = None, debugging = False):
    # view_cols = ['CEP', 'DESC_RUA', 'NUMERO_RUA', 'NUMERO_RUA_iptu', 'checked_number']
    possible_tgts = df_tgts.copy()

    # Correspondencias de numero
    possible_tgts['checked_number'] = np.where(
        possible_tgts['NUMERO_RUA'] == possible_tgts['NUMERO_RUA_iptu'],
        True,
        False
        )

    # Remove casos de lados opostos da rua
    possible_tgts = possible_tgts[(possible_tgts['NUMERO_RUA'] - possible_tgts['NUMERO_RUA_iptu']) % 2 == 0]

    # Indice de diferença dos numeros e relação com testada do predio
    possible_tgts['indice_numero__testada'] = possible_tgts['TOL_NUM'] / possible_tgts['TESTADA PARA CALCULO']

    ###### possible_tgts[view_cols].drop_duplicates()

    # Tratar os casos que não teve correspondencia direta
    aux_del = possible_tgts[possible_tgts['NUMERO_RUA'] != possible_tgts['NUMERO_RUA_iptu']].drop(
        columns = 'checked_number').copy()
    aux_del = pd.merge(aux_del, possible_tgts[possible_tgts['checked_number'] == True][
        ['CEP', 'NUMERO_RUA_iptu', 'checked_number']].drop_duplicates(), on = ['CEP', 'NUMERO_RUA_iptu'],
                       how = 'left').fillna(
        False)

    # Filtra os casos que não tiveram correspondencia
    aux_del = aux_del[aux_del['checked_number'] == False]

    # Filtrar a correspondencia que teve menor diferença entre numeros (mais proxima)
    select_tolerable_iptu = aux_del.sort_values(by = ['CEP', 'DESC_RUA', 'NUMERO_RUA', 'TOL_NUM'])[
        ['CEP', 'DESC_RUA', 'NUMERO_RUA_iptu']].drop_duplicates()
    aux_tolerable = aux_del.loc[select_tolerable_iptu.index]

    # Filtrar a correspondencia que teve menor diferença entre numeros (mais proxima)
    select_tolerable_cg = aux_tolerable.sort_values(by = ['CEP', 'DESC_RUA', 'NUMERO_RUA', 'TOL_NUM'])[
        ['CEP', 'DESC_RUA', 'NUMERO_RUA']].drop_duplicates()
    aux_tolerable_cg = aux_tolerable.loc[select_tolerable_cg.index].sort_values(by = ['CEP', 'NUMERO_RUA'])
    selected_tgts = aux_tolerable_cg.sort_values(by = ['CEP', 'NUMERO_RUA'])

    ###### selected_tgts.head()

    # Possíveis erros -> diferença de numeração muito grande em relação a testada do edificio
    possible_tgts_error = selected_tgts[
        (selected_tgts['TOL_NUM'] > tol_TESTADA * selected_tgts['TESTADA PARA CALCULO']) &
        (selected_tgts['TESTADA PARA CALCULO'] != 0)]

    if scrap_tgts is not None:
        possible_tgts_scrap = scrap_tgts
    else:
        # @Scrap se necessário - Googlemaps
        ## Procura Nome empreendimento + Rua + Numero dos casos em duvida

        # possible_tgts_scrap = pd.merge(possible_tgts_error, df_LOGRADOURO[['Empreendimento_Nome', 'DESC_RUA', 'NUMERO_RUA']],
        #                                on = ['DESC_RUA', 'NUMERO_RUA'],
        #                                how = 'left').drop_duplicates()

        possible_tgts_scrap = possible_tgts_error[
            ['Empreendimento_Nome', 'DESC_RUA', 'NUMERO_RUA_iptu']].drop_duplicates()
        possible_tgts_scrap['query'] = possible_tgts_scrap[possible_tgts_scrap.columns.tolist()].astype(str).agg(
            ' , '.join, axis = 1) + ', São Paulo, SP'

        result = scrape_ceps_from_queries(possible_tgts_scrap['query'].tolist())

        resultados_endereco = pd.DataFrame(result, columns = ['query', 'response'])
        # resultados_endereco.loc[~resultados_endereco['response'].str.contains('SP, '), 'response'] = 'SP, Error'
        resultados_endereco[['Endereco', 'CEP']] = resultados_endereco['response'].str.split(', São Paulo - SP, ',
                                                                                             expand = True)
        resultados_endereco[['DESC_RUA', 'NUMERO_RUA_scrap']] = resultados_endereco['Endereco'].str.extract(
            r'([^\d]+)(\d+)', expand = True)

        resultados_endereco['DESC_RUA'] = resultados_endereco['DESC_RUA'].str.strip()
        resultados_endereco = resultados_endereco[['query', 'NUMERO_RUA_scrap', 'CEP']]

        possible_tgts_scrap = pd.merge(possible_tgts_scrap, resultados_endereco, on = 'query')
        possible_tgts_scrap.to_csv(filename_scrap, index = False)

    if 'NomeEmpreendimento' in possible_tgts_scrap.columns:
        possible_tgts_scrap.rename(columns = {'NomeEmpreendimento': 'Empreendimento_Nome'}, inplace = True)

    aux_scrap = pd.merge(possible_tgts_error, possible_tgts_scrap[['Empreendimento_Nome', 'CEP', 'NUMERO_RUA_scrap']],
                         on = ['CEP', 'Empreendimento_Nome'], how = 'left').drop_duplicates()

    aux_scrap['Aceitar'] = np.where(aux_scrap['NUMERO_RUA_iptu'] == aux_scrap['NUMERO_RUA_scrap'], True, False)

    print(f"Aceitos: {aux_scrap[aux_scrap['Aceitar'] == 1].shape[0]}")
    print(f"Rejeitados: {aux_scrap[aux_scrap['Aceitar'] == 0].shape[0]}")

    if column_order is None:
        column_order = ['Empreendimento_Nome', 'Padrao', 'CEP', 'DESC_RUA', 'NUMERO_RUA',
                        'NUMERO_RUA_iptu', 'NUMERO_RUA_scrap',
                        'Imovel_residencial', 'Imovel_vertical', 'Andares_tipo',
                        'QUANTIDADE DE PAVIMENTOS',
                        'Total_Unidades', 'CONTAGEM_UNIDADES', 'Blocos', 'Unidades_andar',
                        'Idade_predio', 'Fator_obsolescencia_calculado', 'FATOR DE OBSOLESCENCIA',
                        'indice_numero__testada', 'TESTADA PARA CALCULO',
                        'Elevadores', 'Coberturas', 'Dormitorios',
                        'Banheiros_Sociais', 'Vagas_Garagem', 'M2_util_unidade_tipo',
                        'M2_total_unidade_tipo', 'M2_Terreno', 'M2_util_cobertura',
                        'M2_total_cobertura', 'RS_total_unidade_tipo_IGPM',
                        'RS_por_M2_area_util_IGPM', 'RS_por_M2_area_total_IGPM',
                        'RS_total_cobertura_IGPM', 'VALOR DO M2 DO TERRENO', 'AREA DO TERRENO',
                        'VALOR DO M2 DE CONSTRUCAO', 'AREA CONSTRUIDA',
                        'TOL_NUM', 'checked_number',
                        ]

        if debugging == False:
            aux_scrap['NUMERO_RUA'] = aux_scrap['NUMERO_RUA_iptu']
            column_order.remove('NUMERO_RUA_iptu')

    if return_accepted: return aux_scrap[aux_scrap['Aceitar'] == 1][column_order].sort_values(
        by = ['DESC_RUA', 'NUMERO_RUA'])
    return aux_scrap[column_order].sort_values(by = ['DESC_RUA', 'NUMERO_RUA'])


'''
****//********//********//********//********//********//********//********//********//********//********//****
FUNÇÕES SCRAPPING
****//********//********//********//********//********//********//********//********//********//********//****
'''


# Scrapping para obter CEP dada query -> Googlemaps
def scrape_ceps_from_queries(query_list):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the WebDriver (adjust the path to where your ChromeDriver is located)
    service = Service('C:\\Users\\arthu\\USPy\\chromeDriver\\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    results = []
    count_err = 1
    count = 0

    # Iterate through the list of queries
    for idx, query in enumerate(query_list):
        try:
            print(f"Processing {idx+1}/{len(query_list)}: {query}")

            # Search for the query in Google Maps
            search_url = f"https://www.google.com/maps/search/{query}"
            driver.get(search_url)

            # Wait for the results to load
            time.sleep(3)  # Adjust wait time if necessary

            # Locate the address information (span tag with class 'DkEaL')
            address_info = driver.find_element(By.CLASS_NAME, 'Io6YTe').text

            # Append the result (query, address_info)
            results.append((query, address_info))

            # Print the extracted information
            print(f"Extracted: {address_info}")

        except Exception as e:
            print(f"{count_err} / {count}- Error processing {query}")
            results.append((query, "Error"))
            count_err += 1

        count += 1

    # Close the browser after scraping
    driver.quit()

    return results


'''
****//********//********//********//********//********//********//********//********//********//********//****
FUNÇÕES IPTU
****//********//********//********//********//********//********//********//********//********//********//****
'''


# Converter coluna 'TIPO DE PADRAO DA CONSTRUCAO'
## Imovel residencial/comercial
## Imovel vertical/horizontal
## Padrão de construção

# def convert_padrao_iptu(data, drop_original_col = True, compute_obsolence = False):
#     # Create a new column for the remaining part ('padrão X')
#     data['Padrao'] = data['TIPO DE PADRAO DA CONSTRUCAO'].str.extract(r'(padrão \w)')
#     data['Padrao'] = data['Padrao'].str.replace('padrão ', '')
#
#     data['Imovel_residencial'] = np.where(data['TIPO DE PADRAO DA CONSTRUCAO'].str.lower().str.contains('residencial'),
#                                           1, 0)
#
#     # Create 'Imovel_vertical' column: 1 if 'vertical' is in the string, 0 otherwise
#     data['Imovel_vertical'] = np.where(data['TIPO DE PADRAO DA CONSTRUCAO'].str.lower().str.contains('vertical'), 1, 0)
#
#     if drop_original_col:
#         data.drop(columns = 'TIPO DE PADRAO DA CONSTRUCAO', inplace = True)
#     if compute_obsolence:
#         # Apply the function to your DataFrame
#         data['Fator_obsolescencia_calculado'] = data.apply(
#             lambda row: get_obsolescence(row['Idade_predio'], row['Padrao']), axis = 1)
#     return data

def convert_padrao_iptu(data, drop_original_col = True, compute_obsolence = False):
    # Create a new column for the remaining part ('padrão X')
    data['Padrao'] = data['TIPO DE PADRAO DA CONSTRUCAO'].str.extract(r'(padrão \w)')
    data['Padrao'] = data['Padrao'].str.replace('padrão ', '')

#     data['Imovel_residencial'] = np.where(data['TIPO DE PADRAO DA CONSTRUCAO'].str.lower().str.contains('residencial'),
#                                           1, 0)
#
#     # Create 'Imovel_vertical' column: 1 if 'vertical' is in the string, 0 otherwise
#     data['Imovel_vertical'] = np.where(data['TIPO DE PADRAO DA CONSTRUCAO'].str.lower().str.contains('vertical'), 1, 0)
#


    if drop_original_col:
        data.drop(columns = 'TIPO DE PADRAO DA CONSTRUCAO', inplace = True)
    if compute_obsolence:
        # Apply the function to your DataFrame
        data['Fator_obsolescencia_calculado'] = data.apply(
            lambda row: get_obsolescence(row['Idade_predio'], row['Padrao']), axis = 1)
    return data


# Calcular fator de obsolencia
def get_obsolescence(age, pattern):
    # DataFrame containing obsolescence factors
    df_factors = pd.DataFrame({
        'Idade do Prédio (em anos)': ["Menor que 1"] + list(range(1, 43)),
        'Fatores de Obsolescência para padrões A e B': [1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.90, 0.89,
                                                        0.88, 0.86, 0.84, 0.83, 0.81, 0.79, 0.78, 0.76, 0.74, 0.72,
                                                        0.70, 0.68, 0.66, 0.64, 0.62, 0.59, 0.57, 0.55, 0.52, 0.50,
                                                        0.48, 0.45, 0.42, 0.40, 0.37, 0.34, 0.32, 0.29, 0.26, 0.23,
                                                        0.20, 0.20, 0.20],
        'Fatores de Obsolescência para demais padrões e tipos': [1.00, 0.99, 0.99, 0.98, 0.97, 0.96, 0.96, 0.95, 0.94,
                                                                 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.88, 0.86, 0.85,
                                                                 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.76, 0.75,
                                                                 0.74, 0.73, 0.71, 0.70, 0.69, 0.67, 0.66, 0.64, 0.63,
                                                                 0.62, 0.60, 0.59, 0.57, 0.56, 0.54, 0.52]
        })
    # Map age to lookup value
    if age < 1:
        age_lookup = "Menor que 1"
    else:
        age_lookup = age

    # Determine the column to use based on the pattern
    if pattern in ['A', 'B']:
        factor_column = 'Fatores de Obsolescência para padrões A e B'
    else:
        factor_column = 'Fatores de Obsolescência para demais padrões e tipos'

    # Perform the lookup and handle missing results
    factors = df_factors[df_factors['Idade do Prédio (em anos)'] == age_lookup]

    if not factors.empty:
        factor = factors[factor_column].values[0]
    else:
        # Handle the case where no matching age is found (you can adjust this as needed)
        factor = -1  # Or set a default factor value

    return factor


'''
****//********//********//********//********//********//********//********//********//********//********//****
FUNÇÕES DIVERSAS
****//********//********//********//********//********//********//********//********//********//********//****
'''


# COrrigir padrao CEP
def correct_cep(data_raw, apply_standard = True):
    data = data_raw.copy()
    # Step 1: Filter rows where 'CEP' length is 4 and drop duplicates
    # df_GEO_endereco = df_GEO_merged[['NomeEmpreendimento', 'DESC_RUA', 'NUMERO_RUA', 'CEP', 'CEP_LATLONG', 'CEP_RUA']].drop_duplicates().copy()

    # Step 2: Safely create 'CEP_cod_lat', adding a check for non-null and sufficient length
    data['CEP_cod_lat'] = data['CEP_LATLONG'].apply(
        lambda x: x[:5] + '-' + x[5:] if isinstance(x, str) and len(x) >= 6 and apply_standard else x)

    # Step 3: Safely create 'CEP_cod_rua' similarly
    data['CEP_cod_rua'] = data['CEP_RUA'].apply(
        lambda x: x[:5] + '-' + x[5:] if isinstance(x, str) and len(x) >= 6 and apply_standard else x)

    # Step 4: Fill 'CEP_me_salve' with 'CEP_LATLONG' first, then fall back on 'CEP_cod_rua'
    data['CEP_me_salve'] = data['CEP_cod_lat']
    data['CEP_me_salve'] = data['CEP_me_salve'].fillna(data['CEP_cod_rua'])

    data['CEP_cod'] = data['CEP'].apply(
        lambda x: x[:5] + '-' + x[5:] if isinstance(x, str) and len(x) >= 6 and apply_standard else np.nan)
    data.drop(columns = 'CEP', inplace = True)

    data['CEP'] = data['CEP_cod'].fillna(data['CEP_me_salve'])

    data['Andares_tipo'] = data['Andares_tipo'].astype(float)
    data['Total_Unidades'] = data['Total_Unidades'].astype(float)

    return data


# Converter datatime
def convert_to_datetime(date_str, format = "%m/%y"):
    if pd.isna(date_str):
        return pd.NaT

    month_map = {
        "JAN": "01", "FEV": "02", "MAR": "03", "ABR": "04",
        "MAI": "05", "JUN": "06", "JUL": "07", "AGO": "08",
        "SET": "09", "OUT": "10", "NOV": "11", "DEZ": "12"
        }

    for month_abbr, month_num in month_map.items():
        if month_abbr in date_str:
            date_str = date_str.replace(month_abbr, month_num)
    return pd.to_datetime(date_str, format = format)


# Concatenar elementos em conjunto
def concatenate_unique(lst1):
    seen = set()
    result = []
    for lst in lst1:
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result
