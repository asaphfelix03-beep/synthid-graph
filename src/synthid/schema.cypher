// Schéma du graphe (Neo4j 5). Uniquement des jetons : aucune PII en clair.
// Les jetons HMAC sont propres à chaque banque : les sous-graphes des banques sont disjoints.
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE;
CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE;
CREATE CONSTRAINT nid_token IF NOT EXISTS FOR (n:NationalID) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT phone_token IF NOT EXISTS FOR (n:Phone) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT email_token IF NOT EXISTS FOR (n:Email) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT address_token IF NOT EXISTS FOR (n:Address) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT device_token IF NOT EXISTS FOR (n:Device) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT ip_token IF NOT EXISTS FOR (n:IP) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT employer_token IF NOT EXISTS FOR (n:Employer) REQUIRE n.token IS UNIQUE;
CREATE CONSTRAINT merchant_token IF NOT EXISTS FOR (n:Merchant) REQUIRE n.token IS UNIQUE;
CREATE INDEX person_bank IF NOT EXISTS FOR (p:Person) ON (p.bank);
CREATE INDEX person_risk IF NOT EXISTS FOR (p:Person) ON (p.risk_score);
