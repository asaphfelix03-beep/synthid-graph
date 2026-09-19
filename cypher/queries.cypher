// Requêtes d'investigation (Neo4j Browser : http://localhost:7474).
// Toutes les relations portent first_day / last_day : ajouter « WHERE r.first_day <= $T » pour une vue « as-of T ».

// 1. Identités partageant ≥ 2 attributs forts, super-nœuds (CGNAT, résidences, gros employeurs) exclus
MATCH (p1:Person)-[:HAS_NATIONAL_ID|HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE]->(a)
      <-[:HAS_NATIONAL_ID|HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE]-(p2:Person)
WHERE p1.person_id < p2.person_id
  AND COUNT { (a)<--(:Person) } <= 20
WITH p1, p2, collect(DISTINCT labels(a)[0]) AS shared
WHERE size(shared) >= 2
RETURN p1.person_id, p2.person_id, shared
ORDER BY size(shared) DESC LIMIT 50;

// 2. Un numéro d'identité national porté par plusieurs identités (nom + date de naissance différents)
MATCH (n:NationalID)<-[:HAS_NATIONAL_ID]-(p:Person)
WITH n, collect(DISTINCT p.id_token) AS identities, collect(p.person_id) AS persons
WHERE size(identities) >= 2
RETURN n.token, size(identities) AS n_identities, persons
ORDER BY n_identities DESC LIMIT 25;

// 3. Voisinage à 2 sauts d'une fraude confirmée (contagion) : clients encore ouverts à surveiller
MATCH (f:Person {known_fraud: true})-[:HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE|USED_IP]->(a)<-[]-(p:Person)
WHERE p.known_fraud = false AND COUNT { (a)<--(:Person) } <= 20
RETURN p.person_id, collect(DISTINCT labels(a)[0]) AS via, count(DISTINCT f) AS n_fraud_neighbours
ORDER BY n_fraud_neighbours DESC LIMIT 50;

// 4. Vendeurs de « tradelines » : comptes hôtes d'un nombre anormal d'utilisateurs autorisés
MATCH (p:Person)-[:AUTHORIZED_USER_ON]->(a:Account)<-[:OWNS]-(host:Person)
WITH host, count(DISTINCT p) AS dependents
WHERE dependents >= 3
RETURN host.person_id, dependents ORDER BY dependents DESC;

// 5. Dossier d'alerte : sous-graphe d'une communauté suspecte (après write-back des scores)
MATCH (p:Person {community_alert: $community_id})-[r]->(a)
RETURN p, r, a;

// 6. Graph Data Science : projection des identités et de leurs attributs, puis WCC et Louvain
CALL gds.graph.project('identity',
  ['Person', 'Phone', 'Email', 'Address', 'Device', 'NationalID'],
  {HAS_PHONE: {orientation: 'UNDIRECTED'}, HAS_EMAIL: {orientation: 'UNDIRECTED'},
   LIVES_AT: {orientation: 'UNDIRECTED'}, USED_DEVICE: {orientation: 'UNDIRECTED'},
   HAS_NATIONAL_ID: {orientation: 'UNDIRECTED'}});
CALL gds.wcc.stream('identity') YIELD nodeId, componentId
WITH componentId, count(*) AS size WHERE size > 5
RETURN componentId, size ORDER BY size DESC LIMIT 20;
