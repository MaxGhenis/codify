import { capitalize } from '../vendor/tbi-ui';

const GENERIC: Record<string, string> = {
  act: 'Act',
  bill: 'Bill',
  ordinance: 'Ordinance',
  decree: 'Decree',
  regulation: 'Regulation',
  treaty: 'Treaty',
};

const BY_JURISDICTION: Record<string, Record<string, string>> = {
  gb: {
    ukpga: 'UK Public General Act',
    ukla: 'UK Local Act',
    asp: 'Scottish Parliament Act',
    anaw: 'National Assembly for Wales Act',
    asc: 'Senedd Act',
    nia: 'Northern Ireland Assembly Act',
    aosp: 'Old Scottish Parliament Act',
    aep: 'England Act',
    apgb: 'Great Britain Act',
    aip: 'Ireland Act',
    mwa: 'Welsh Assembly Measure',
    ukcm: 'Church Measure',
    mnia: 'Northern Ireland Assembly Measure',
    apni: 'Northern Ireland Parliament Act',
    gbla: 'Great Britain Local Act',
    ukppa: 'UK Private or Personal Act',
    uksi: 'UK Statutory Instrument',
    wsi: 'Welsh Statutory Instrument',
    ssi: 'Scottish Statutory Instrument',
    nisr: 'Northern Ireland Statutory Rule',
    nisi: 'Northern Ireland Order in Council',
    ukmd: 'UK Ministerial Direction',
    nisro: 'Northern Ireland Statutory Rule or Order',
    uksro: 'UK Statutory Rule or Order',
    ukmo: 'UK Ministerial Order',
    ukci: 'Church Instrument',
  },
};

export function formatDoctype(d: string, jurisdiction?: string): string {
  return (jurisdiction && BY_JURISDICTION[jurisdiction]?.[d]) ?? GENERIC[d] ?? capitalize(d);
}
