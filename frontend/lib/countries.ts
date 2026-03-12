/**
 * Full ISO 3166-1 alpha-2 country list via Intl.DisplayNames.
 * Zero dependencies — uses the browser's built-in locale API.
 *
 * Also exported: GLOBAL_OPTION for "worldwide / no country filter".
 * Backend accepts: ISO alpha-2 code (e.g. "IN", "US") or "global".
 */

export const GLOBAL_OPTION = { code: "global", name: "Global" };

export const ALL_COUNTRIES: { code: string; name: string }[] = (() => {
  const codes = [
    "AF","AL","DZ","AD","AO","AG","AR","AM","AU","AT","AZ","BS","BH","BD","BB",
    "BY","BE","BZ","BJ","BT","BO","BA","BW","BR","BN","BG","BF","BI","KH","CM",
    "CA","CV","CF","TD","CL","CN","CO","KM","CG","CR","HR","CU","CY","CZ","CD",
    "DK","DJ","DM","DO","EC","EG","SV","GQ","ER","EE","SZ","ET","FJ","FI","FR",
    "GA","GM","GE","DE","GH","GR","GD","GT","GN","GW","GY","HT","HN","HU","IS",
    "IN","ID","IR","IQ","IE","IL","IT","CI","JM","JP","JO","KZ","KE","KI","KW",
    "KG","LA","LV","LB","LS","LR","LY","LI","LT","LU","MG","MW","MY","MV","ML",
    "MT","MH","MR","MU","MX","FM","MD","MC","MN","ME","MA","MZ","MM","NA","NR",
    "NP","NL","NZ","NI","NE","NG","KP","MK","NO","OM","PK","PW","PS","PA","PG",
    "PY","PE","PH","PL","PT","QA","RO","RU","RW","KN","LC","VC","WS","SM","ST",
    "SA","SN","RS","SC","SL","SG","SK","SI","SB","SO","ZA","KR","SS","ES","LK",
    "SD","SR","SE","CH","SY","TW","TJ","TZ","TH","TL","TG","TO","TT","TN","TR",
    "TM","TV","UG","UA","AE","GB","US","UY","UZ","VU","VE","VN","YE","ZM","ZW",
  ];
  try {
    const fmt = new Intl.DisplayNames(["en"], { type: "region" });
    return codes
      .map((c) => ({ code: c, name: fmt.of(c) ?? c }))
      .sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    // SSR / old browser fallback — keep original order, show code as name
    return codes.map((c) => ({ code: c, name: c }));
  }
})();

/** Full list with "Global" pinned at top — use for profile region selects. */
export const REGION_OPTIONS = [GLOBAL_OPTION, ...ALL_COUNTRIES];

/** Resolve a stored region value to a display name. */
export function regionLabel(code: string): string {
  if (!code || code.toLowerCase() === "global") return GLOBAL_OPTION.name;
  const found = ALL_COUNTRIES.find((c) => c.code.toUpperCase() === code.toUpperCase());
  return found?.name ?? code;
}
