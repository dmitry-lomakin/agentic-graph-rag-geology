% Query routing rules — declarative replacement for regex patterns in router.py
% Keywords map query tokens to categories

% Relation keywords (RU + EN)
keyword(/relation, "связ").
keyword(/relation, "отношен").
keyword(/relation, "соедин").
keyword(/relation, "relat").
keyword(/relation, "connect").
keyword(/relation, "link").
keyword(/relation, "between").
keyword(/relation, "между").

% Multi-hop keywords
keyword(/multi_hop, "цепочк").
keyword(/multi_hop, "путь").
keyword(/multi_hop, "сравн").
keyword(/multi_hop, "через").
keyword(/multi_hop, "chain").
keyword(/multi_hop, "path").
keyword(/multi_hop, "compar").
keyword(/multi_hop, "through").
keyword(/multi_hop, "affect").
keyword(/multi_hop, "влия").

% Global keywords
keyword(/global, "все").
keyword(/global, "кажд").
keyword(/global, "обзор").
keyword(/global, "список").
keyword(/global, "all").
keyword(/global, "every").
keyword(/global, "overview").
keyword(/global, "list").
keyword(/global, "summar").

% Temporal keywords
keyword(/temporal, "когда").
keyword(/temporal, "дата").
keyword(/temporal, "время").
keyword(/temporal, "истори").
keyword(/temporal, "when").
keyword(/temporal, "date").
keyword(/temporal, "timeline").
keyword(/temporal, "before").
keyword(/temporal, "after").
keyword(/temporal, "до").
keyword(/temporal, "после").

% --- Geology domain keywords ---

% Regulatory keywords (ГКЗ, standards, laws)
keyword(/regulatory, "гкз").
keyword(/regulatory, "государственн").
keyword(/regulatory, "экспертиз").
keyword(/regulatory, "закон о недрах").
keyword(/regulatory, "кондиции").
keyword(/regulatory, "классификация запасов").
keyword(/regulatory, "форма 5-гр").
keyword(/regulatory, "тэо кондиций").
keyword(/regulatory, "роснедра").
keyword(/regulatory, "гост").
keyword(/regulatory, "приказ мпр").
keyword(/regulatory, "методическ").
keyword(/regulatory, "нормативн").

% Software comparison keywords
keyword(/comparison, "micromine").
keyword(/comparison, "surpac").
keyword(/comparison, "datamine").
keyword(/comparison, "leapfrog").
keyword(/comparison, "vulcan").
keyword(/comparison, "геомикс").
keyword(/comparison, "digimine").
keyword(/comparison, "ггис").
keyword(/comparison, "сравн").
keyword(/comparison, "отличи").
keyword(/comparison, "преимущ").
keyword(/comparison, "vs").
keyword(/comparison, "versus").
keyword(/comparison, "альтернатив").

% Geology domain lookup keywords
keyword(/geology_lookup, "кригинг").
keyword(/geology_lookup, "kriging").
keyword(/geology_lookup, "вариограмм").
keyword(/geology_lookup, "variogram").
keyword(/geology_lookup, "блочн").
keyword(/geology_lookup, "block model").
keyword(/geology_lookup, "запасов").
keyword(/geology_lookup, "reserve").
keyword(/geology_lookup, "resource estim").
keyword(/geology_lookup, "скважин").
keyword(/geology_lookup, "drillhole").
keyword(/geology_lookup, "drill hole").
keyword(/geology_lookup, "каркасн").
keyword(/geology_lookup, "wireframe").
keyword(/geology_lookup, "триангуляц").
keyword(/geology_lookup, "карьер").
keyword(/geology_lookup, "pit optim").
keyword(/geology_lookup, "mine plan").
keyword(/geology_lookup, "опробован").
keyword(/geology_lookup, "assay").
keyword(/geology_lookup, "литолог").
keyword(/geology_lookup, "интерполяц").
keyword(/geology_lookup, "подсчёт").
keyword(/geology_lookup, "подсчет").
keyword(/geology_lookup, "рудн").
keyword(/geology_lookup, "ore body").
keyword(/geology_lookup, "геостатист").
keyword(/geology_lookup, "geostatist").

% Match: keyword must bind Word first, then query_contains checks it
match(Query, Category) :- keyword(Category, Word), query_contains(Query, Word).

% Tool mapping per category
tool_for(/simple, "vector_search").
tool_for(/relation, "cypher_traverse").
tool_for(/multi_hop, "cypher_traverse").
tool_for(/global, "full_document_read").
tool_for(/temporal, "temporal_query").
tool_for(/regulatory, "vector_search").
tool_for(/comparison, "comprehensive_search").
tool_for(/geology_lookup, "vector_search").

% Route: if any category matches, use its tool
route_to(Tool, Query) :- match(Query, Category), tool_for(Category, Tool).

% Default: no match → vector_search
route_to("vector_search", Query) :- current_query(Query), !match(Query, X).
