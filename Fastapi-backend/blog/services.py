from sqlalchemy.orm import Session
from models import Poster

def get_customer_info_by_contrat(contrat: int, db: Session):
    # Query Poster table for given contrat
    posters = db.query(Poster).filter(Poster.Contrat == contrat).all()

    # If no records, return empty list
    if not posters:
        return []

    # Convert to list of dicts for easy processing
    bills = []
    for p in posters:
        bills.append({
            "numero_facture": p.Numero_Facture,
            "contrat": p.Contrat,
            "consommation": p.Consommation,
            "montant_ttc": p.Montant_TTC,
            "paiement_caisse": p.Paiement_Caisse,
            "type_facture": p.Type_Facture,
            "ref_geo": p.Ref_Geo,
            "date_de_facturation": p.Date_de_Facturation.isoformat() if p.Date_de_Facturation else None,
            "type_de_facturation": p.Type_de_Facturation
        })

    return bills
