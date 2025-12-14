"""CMS-specific extraction strategies for DumpAI.

Each CMS has known tables with admin credentials.
When CMS is detected, we skip enumeration and dump directly.

Reference: https://github.com/ipa-lab/hackingBuddyGPT
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CMSTable:
    """Table definition with extraction priority."""
    name: str
    columns: List[str]
    priority: int = 1  # 1=critical, 2=high, 3=medium
    description: str = ""
    prefix_aware: bool = True  # Table name uses CMS prefix


@dataclass
class CMSStrategy:
    """Extraction strategy for a specific CMS."""
    name: str
    detection_tables: List[str]  # Tables that indicate this CMS
    detection_patterns: List[str]  # Patterns in table names
    default_prefix: str
    tables: Dict[str, List[CMSTable]]  # Category -> Tables
    admin_url_patterns: List[str] = field(default_factory=list)
    notes: str = ""


# PrestaShop Strategy
PRESTASHOP = CMSStrategy(
    name="PrestaShop",
    detection_tables=["ps_employee", "ps_shop", "ps_configuration"],
    detection_patterns=["ps_", "prestashop"],
    default_prefix="ps_",
    admin_url_patterns=["/admin", "/admin-*", "/backoffice"],
    tables={
        "user_data": [
            CMSTable(
                name="employee",
                columns=["id_employee", "email", "passwd", "firstname", "lastname", "active", "id_profile", "last_connection_date"],
                priority=1,
                description="Back-office administrators"
            ),
            CMSTable(
                name="webservice_account",
                columns=["id_webservice_account", "key", "active", "description"],
                priority=1,
                description="API keys with full access"
            ),
        ],
        "customer_data": [
            CMSTable(
                name="customer",
                columns=["id_customer", "email", "passwd", "firstname", "lastname", "active"],
                priority=2,
                description="Customer accounts"
            ),
            CMSTable(
                name="address",
                columns=["id_address", "id_customer", "address1", "address2", "city", "phone", "phone_mobile"],
                priority=3,
                description="Customer addresses"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="configuration",
                columns=["id_configuration", "name", "value"],
                priority=2,
                description="System config (may contain keys)"
            ),
            CMSTable(
                name="shop_url",
                columns=["id_shop_url", "domain", "domain_ssl", "physical_uri"],
                priority=3,
                description="Shop domains"
            ),
        ],
        "api_key": [
            CMSTable(
                name="webservice_account",
                columns=["id_webservice_account", "key", "active", "description"],
                priority=1,
                description="PrestaShop API keys"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="employee",
                columns=["email", "passwd"],
                priority=1,
                description="Admin email+password"
            ),
            CMSTable(
                name="customer",
                columns=["email", "passwd"],
                priority=2,
                description="Customer email+password"
            ),
        ],
    },
    notes="Prefix ps_ is customizable. Check configuration table for secrets."
)


# WordPress / WooCommerce Strategy
WORDPRESS = CMSStrategy(
    name="WordPress",
    detection_tables=["wp_users", "wp_posts", "wp_options"],
    detection_patterns=["wp_", "wordpress"],
    default_prefix="wp_",
    admin_url_patterns=["/wp-admin", "/wp-login.php"],
    tables={
        "user_data": [
            CMSTable(
                name="users",
                columns=["ID", "user_login", "user_pass", "user_email", "user_registered", "display_name"],
                priority=1,
                description="All users including admins"
            ),
            CMSTable(
                name="usermeta",
                columns=["umeta_id", "user_id", "meta_key", "meta_value"],
                priority=1,
                description="User roles and capabilities (filter: wp_capabilities)"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="options",
                columns=["option_id", "option_name", "option_value"],
                priority=2,
                description="Site config (may contain API keys, secrets)"
            ),
        ],
        "api_key": [
            CMSTable(
                name="options",
                columns=["option_id", "option_name", "option_value"],
                priority=2,
                description="API keys in options table"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="users",
                columns=["user_login", "user_pass", "user_email"],
                priority=1,
                description="User email+password"
            ),
        ],
        "customer_data": [
            CMSTable(
                name="usermeta",
                columns=["user_id", "meta_key", "meta_value"],
                priority=2,
                description="User metadata (billing info for WooCommerce)"
            ),
        ],
    },
    notes="WooCommerce uses same tables. Check usermeta for roles."
)


# OpenCart Strategy
OPENCART = CMSStrategy(
    name="OpenCart",
    detection_tables=["oc_user", "oc_product", "oc_setting"],
    detection_patterns=["oc_", "opencart"],
    default_prefix="oc_",
    admin_url_patterns=["/admin"],
    tables={
        "user_data": [
            CMSTable(
                name="user",
                columns=["user_id", "username", "password", "salt", "email", "status", "user_group_id"],
                priority=1,
                description="Admin panel users"
            ),
        ],
        "customer_data": [
            CMSTable(
                name="customer",
                columns=["customer_id", "email", "password", "salt", "firstname", "lastname", "telephone"],
                priority=2,
                description="Store customers"
            ),
            CMSTable(
                name="address",
                columns=["address_id", "customer_id", "firstname", "lastname", "address_1", "city", "postcode"],
                priority=3,
                description="Customer addresses"
            ),
        ],
        "api_key": [
            CMSTable(
                name="api",
                columns=["api_id", "username", "key", "status"],
                priority=1,
                description="OpenCart API keys"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="setting",
                columns=["setting_id", "store_id", "code", "key", "value"],
                priority=2,
                description="Store settings"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="user",
                columns=["email", "password"],
                priority=1,
                description="Admin email+password"
            ),
            CMSTable(
                name="customer",
                columns=["email", "password"],
                priority=2,
                description="Customer email+password"
            ),
        ],
    },
    notes="Salt column exists in older versions. Prefix oc_ often customized."
)


# Joomla Strategy
JOOMLA = CMSStrategy(
    name="Joomla",
    detection_tables=["jos_users", "joomla_users"],
    detection_patterns=["jos_", "joomla", "_users", "_content"],
    default_prefix="jos_",
    admin_url_patterns=["/administrator"],
    tables={
        "user_data": [
            CMSTable(
                name="users",
                columns=["id", "name", "username", "email", "password", "block", "registerDate", "lastvisitDate"],
                priority=1,
                description="All users"
            ),
            CMSTable(
                name="user_usergroup_map",
                columns=["user_id", "group_id"],
                priority=2,
                description="User group mapping (admin = group 8)"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="extensions",
                columns=["extension_id", "name", "type", "params"],
                priority=3,
                description="Extensions config"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="users",
                columns=["email", "password", "username"],
                priority=1,
                description="User email+password"
            ),
        ],
        "api_key": [
            CMSTable(
                name="extensions",
                columns=["extension_id", "name", "params"],
                priority=2,
                description="Extension params may contain API keys"
            ),
        ],
    },
    notes="Dynamic prefix (default jos_). Admin group ID typically 7 or 8."
)


# Magento Strategy
MAGENTO = CMSStrategy(
    name="Magento",
    detection_tables=["admin_user", "catalog_product_entity", "eav_attribute"],
    detection_patterns=["admin_user", "catalog_", "eav_"],
    default_prefix="",
    admin_url_patterns=["/admin", "/admin_*", "/backend"],
    tables={
        "user_data": [
            CMSTable(
                name="admin_user",
                columns=["user_id", "username", "password", "email", "is_active", "created", "lognum"],
                priority=1,
                prefix_aware=False,
                description="Magento admin users"
            ),
        ],
        "customer_data": [
            CMSTable(
                name="customer_entity",
                columns=["entity_id", "email", "password_hash", "created_at", "is_active"],
                priority=2,
                prefix_aware=False,
                description="Customer accounts"
            ),
        ],
        "api_key": [
            CMSTable(
                name="oauth_token",
                columns=["entity_id", "token", "secret", "type"],
                priority=1,
                prefix_aware=False,
                description="OAuth tokens"
            ),
            CMSTable(
                name="integration",
                columns=["integration_id", "name", "email", "consumer_id", "status"],
                priority=2,
                prefix_aware=False,
                description="API integrations"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="core_config_data",
                columns=["config_id", "scope", "path", "value"],
                priority=2,
                prefix_aware=False,
                description="System config (may contain keys)"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="admin_user",
                columns=["email", "password", "username"],
                priority=1,
                prefix_aware=False,
                description="Admin email+password"
            ),
            CMSTable(
                name="customer_entity",
                columns=["email", "password_hash"],
                priority=2,
                prefix_aware=False,
                description="Customer email+password"
            ),
        ],
    },
    notes="Magento 1 uses admin_user. Magento 2 same structure."
)


# Drupal Strategy
DRUPAL = CMSStrategy(
    name="Drupal",
    detection_tables=["users_field_data", "node", "field_config"],
    detection_patterns=["users_field", "node__", "field_"],
    default_prefix="",
    admin_url_patterns=["/user", "/admin"],
    tables={
        "user_data": [
            CMSTable(
                name="users_field_data",
                columns=["uid", "name", "pass", "mail", "status", "created", "access", "login"],
                priority=1,
                prefix_aware=False,
                description="User accounts (Drupal 8/9/10)"
            ),
            CMSTable(
                name="users",
                columns=["uid", "name", "pass", "mail", "status", "created"],
                priority=1,
                prefix_aware=False,
                description="User accounts (Drupal 7)"
            ),
            CMSTable(
                name="user__roles",
                columns=["entity_id", "roles_target_id"],
                priority=2,
                prefix_aware=False,
                description="User roles mapping"
            ),
        ],
        "email_pass": [
            CMSTable(
                name="users_field_data",
                columns=["name", "pass", "mail"],
                priority=1,
                prefix_aware=False,
                description="User email+password (Drupal 8+)"
            ),
            CMSTable(
                name="users",
                columns=["name", "pass", "mail"],
                priority=1,
                prefix_aware=False,
                description="User email+password (Drupal 7)"
            ),
        ],
        "sys_data": [
            CMSTable(
                name="config",
                columns=["collection", "name", "data"],
                priority=2,
                prefix_aware=False,
                description="Configuration storage"
            ),
        ],
        "api_key": [
            CMSTable(
                name="key_value",
                columns=["collection", "name", "value"],
                priority=2,
                prefix_aware=False,
                description="Key-value storage (may contain tokens)"
            ),
        ],
    },
    notes="Drupal 7 uses 'users', Drupal 8+ uses 'users_field_data'."
)


# All strategies
CMS_STRATEGIES: Dict[str, CMSStrategy] = {
    "prestashop": PRESTASHOP,
    "wordpress": WORDPRESS,
    "woocommerce": WORDPRESS,  # Same as WordPress
    "opencart": OPENCART,
    "joomla": JOOMLA,
    "magento": MAGENTO,
    "drupal": DRUPAL,
}


def detect_cms_from_tables(tables: List[str]) -> Optional[str]:
    """Detect CMS from list of table names."""
    tables_lower = [t.lower() for t in tables]
    tables_str = " ".join(tables_lower)
    
    scores = {}
    
    for cms_name, strategy in CMS_STRATEGIES.items():
        score = 0
        
        for detection_table in strategy.detection_tables:
            base_name = detection_table.replace(strategy.default_prefix, "")
            for t in tables_lower:
                if base_name in t or detection_table.lower() in t:
                    score += 10
        
        for pattern in strategy.detection_patterns:
            if pattern.lower() in tables_str:
                score += 5
        
        if score > 0:
            scores[cms_name] = score
    
    if scores:
        best = max(scores, key=lambda x: scores[x])
        if scores[best] >= 10:
            return best
    
    return None


def get_extraction_plan(cms: str, prefix: str, categories: List[str]) -> Dict[str, List[str]]:
    """
    Get extraction plan for CMS.
    
    Returns: {table_name: [columns]}
    """
    strategy = CMS_STRATEGIES.get(cms.lower())
    if not strategy:
        return {}
    
    plan = {}
    
    for category in categories:
        if category not in strategy.tables:
            continue
        
        for cms_table in strategy.tables[category]:
            if cms_table.prefix_aware:
                table_name = f"{prefix}{cms_table.name}"
            else:
                table_name = cms_table.name
            
            if table_name not in plan:
                plan[table_name] = []
            
            for col in cms_table.columns:
                if col not in plan[table_name]:
                    plan[table_name].append(col)
    
    return plan


def get_priority_tables(cms: str, prefix: str, max_priority: int = 2) -> List[str]:
    """Get high-priority tables for quick extraction."""
    strategy = CMS_STRATEGIES.get(cms.lower())
    if not strategy:
        return []
    
    tables = []
    for category_tables in strategy.tables.values():
        for cms_table in category_tables:
            if cms_table.priority <= max_priority:
                if cms_table.prefix_aware:
                    table_name = f"{prefix}{cms_table.name}"
                else:
                    table_name = cms_table.name
                
                if table_name not in tables:
                    tables.append(table_name)
    
    return tables


def detect_prefix(tables: List[str], cms: str) -> str:
    """Detect actual table prefix from table names."""
    strategy = CMS_STRATEGIES.get(cms.lower())
    if not strategy:
        return ""
    
    for detection_table in strategy.detection_tables:
        base_name = detection_table.replace(strategy.default_prefix, "")
        
        for table in tables:
            if table.lower().endswith(base_name):
                prefix = table[:len(table) - len(base_name)]
                if prefix:
                    return prefix
    
    return strategy.default_prefix


def get_cms_info(cms: str) -> Optional[CMSStrategy]:
    """Get CMS strategy info."""
    return CMS_STRATEGIES.get(cms.lower())
